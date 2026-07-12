"""SHAP explainability for the TF-IDF + Logistic Regression AE classifier.

Multinomial Logistic Regression produces one coefficient vector per class
(`coef_` has shape (n_classes, n_features)). Each class's linear
score = coef_[class] . tfidf_vector + intercept_[class] is exactly the
function SHAP's LinearExplainer expects, so we build one linear explainer
per class of interest directly from that coefficient row rather than
treating the whole multiclass model as a black box. This keeps every
attribution mapped to a single, exact linear term instead of an
approximation, and Logistic Regression was chosen for this classifier
specifically so this clean SHAP mapping is possible (see ae_classifier.py).
"""
import sys
from pathlib import Path

import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.ae_classifier import load_model

MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "ae_classifier.joblib"
PLOTS_DIR = PROJECT_ROOT / "models" / "shap_plots"


class AEShapExplainer:
    def __init__(self, pipeline=None):
        self.pipeline = pipeline or load_model(MODEL_PATH)
        self.vectorizer = self.pipeline.named_steps["tfidf"]
        self.clf = self.pipeline.named_steps["clf"]
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.classes = list(self.clf.classes_)

    def _class_linear_explainer(self, class_label: str, background_dense: np.ndarray):
        class_idx = self.classes.index(class_label)
        coef = self.clf.coef_[class_idx]
        intercept = self.clf.intercept_[class_idx]
        return shap.LinearExplainer((coef, intercept), background_dense)

    def _to_dense(self, clean_texts):
        return np.asarray(self.vectorizer.transform(clean_texts).todense())

    def global_importance(self, clean_texts: list, class_label: str = "Severe AE", top_k: int = 15) -> dict:
        """Mean |SHAP value| per feature across a set of notes, for one class."""
        background = self._to_dense(clean_texts[: min(200, len(clean_texts))])
        explainer = self._class_linear_explainer(class_label, background)
        shap_values = explainer(background)

        mean_abs = np.abs(shap_values.values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_k]

        return {
            "class_label": class_label,
            "top_features": [
                {"feature": self.feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 5)}
                for i in top_idx
            ],
            "_shap_values": shap_values,
            "_feature_names": self.feature_names,
        }

    def plot_global_summary(self, clean_texts: list, class_label: str = "Severe AE", save_path: Path = None) -> Path:
        result = self.global_importance(clean_texts, class_label)
        top = result["top_features"]

        fig, ax = plt.subplots(figsize=(7, 5))
        features = [t["feature"] for t in top][::-1]
        values = [t["mean_abs_shap"] for t in top][::-1]
        ax.barh(features, values, color="#4C72B0")
        ax.set_xlabel("Mean |SHAP value| (impact on model output)")
        ax.set_title(f"Global SHAP feature importance — {class_label} classification")
        fig.tight_layout()

        save_path = save_path or (PLOTS_DIR / f"global_summary_{class_label.replace(' ', '_').lower()}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    def local_explanation(self, clean_text: str, class_label: str = None, background_dense: np.ndarray = None) -> dict:
        """Local SHAP explanation for a single note. Explains the predicted
        class by default (or a specified class)."""
        vector = self._to_dense([clean_text])
        if class_label is None:
            probs = self.pipeline.predict_proba([clean_text])[0]
            class_label = self.classes[int(np.argmax(probs))]

        background = background_dense if background_dense is not None else np.zeros((1, len(self.feature_names)))
        explainer = self._class_linear_explainer(class_label, background)
        shap_values = explainer(vector)

        contributions = shap_values.values[0]
        nonzero_idx = np.nonzero(contributions)[0]
        ranked = sorted(nonzero_idx, key=lambda i: -abs(contributions[i]))

        return {
            "class_label": class_label,
            "base_value": float(shap_values.base_values[0]),
            "contributions": [
                {"feature": self.feature_names[i], "shap_value": round(float(contributions[i]), 5)}
                for i in ranked
            ],
            "_explanation": shap.Explanation(
                values=contributions, base_values=shap_values.base_values[0],
                data=vector[0], feature_names=list(self.feature_names),
            ),
        }

    def plot_local_waterfall(self, clean_text: str, class_label: str = None, save_path: Path = None) -> Path:
        result = self.local_explanation(clean_text, class_label=class_label)
        exp = result["_explanation"]

        # Restrict to non-zero features for a readable waterfall
        nz = np.nonzero(exp.values)[0]
        if len(nz) == 0:
            nz = np.arange(min(10, len(exp.values)))
        small_exp = shap.Explanation(
            values=exp.values[nz], base_values=exp.base_values,
            data=exp.data[nz], feature_names=[exp.feature_names[i] for i in nz],
        )

        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(small_exp, show=False, max_display=12)
        plt.title(f"Local SHAP explanation — predicted: {result['class_label']}")
        plt.tight_layout()

        save_path = save_path or (PLOTS_DIR / "local_waterfall_example.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def business_narrative(self, clean_text: str, class_label: str = None, top_k: int = 3) -> str:
        result = self.local_explanation(clean_text, class_label=class_label)
        positive = [c for c in result["contributions"] if c["shap_value"] > 0][:top_k]

        if not positive:
            return f"No strong individual words pushed this note toward '{result['class_label']}' classification."

        words = [f"'{c['feature']}'" for c in positive]
        if len(words) == 1:
            word_str = words[0]
        else:
            word_str = ", ".join(words[:-1]) + f" and {words[-1]}"

        return (
            f"The word{'s' if len(words) > 1 else ''} {word_str} "
            f"{'were' if len(words) > 1 else 'was'} the strongest predictor{'s' if len(words) > 1 else ''} "
            f"of {result['class_label']} classification in this note."
        )
