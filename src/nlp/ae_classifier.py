"""TF-IDF + Logistic Regression adverse-event classifier.

Classifies a clinical note into one of three severity classes:
No AE / Mild AE / Severe AE. Logistic Regression is chosen over a
higher-capacity model (e.g. Random Forest/XGBoost) specifically because
(a) the feature space is a sparse bag-of-lemmas where a linear decision
boundary is already close to Bayes-optimal, and (b) its coefficients are
directly interpretable and pair cleanly with SHAP's linear explainer for
both global and local explanations — a requirement for this project.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

CLASSES = ["No AE", "Mild AE", "Severe AE"]


def build_pipeline(max_features: int = 2000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train_test_split_stratified(X, y, test_size: float = 0.2, seed: int = 42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASSES,
    }


def save_model(pipeline: Pipeline, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)


def predict_with_confidence(pipeline: Pipeline, texts: list) -> list:
    probs = pipeline.predict_proba(texts)
    preds = pipeline.classes_[np.argmax(probs, axis=1)]
    confidences = np.max(probs, axis=1)
    return [
        {"prediction": pred, "confidence": round(float(conf), 4), "class_probabilities": dict(zip(pipeline.classes_, p.round(4)))}
        for pred, conf, p in zip(preds, confidences, probs)
    ]
