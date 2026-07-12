"""Single-note inference helper: given raw clinical note text, run the full
NLP pipeline (preprocess -> classify -> extract entities) and return a
structured result for the Streamlit "Clinical Notes Analyser" page.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.preprocessing import preprocess_text
from src.nlp.ner import MedicalNER, severity_score
from src.nlp.ae_classifier import load_model

MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "ae_classifier.joblib"

_ner = None
_model = None


def _get_ner():
    global _ner
    if _ner is None:
        _ner = MedicalNER()
    return _ner


def _get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def analyze_note(raw_text: str) -> dict:
    ner = _get_ner()
    model = _get_model()

    entities = ner.extract(raw_text)
    clean_text = preprocess_text(raw_text, nlp=ner.nlp)

    probs = model.predict_proba([clean_text])[0]
    classes = model.classes_
    pred_idx = probs.argmax()
    prediction = classes[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "raw_text": raw_text,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "class_probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "entities": entities,
        "severity_score_0_3": severity_score(entities),
        "clean_text": clean_text,
    }


if __name__ == "__main__":
    sample = "Moderate dizziness reported. Dose reduced from 20mg to 10mg. Follow-up scheduled in 2 weeks."
    import json
    print(json.dumps(analyze_note(sample), indent=2))
