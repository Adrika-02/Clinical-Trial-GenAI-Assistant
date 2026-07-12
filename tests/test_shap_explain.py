import numpy as np

from src.nlp.ae_classifier import build_pipeline
from src.explainability.shap_explain import AEShapExplainer


def _toy_pipeline():
    texts = [
        "no adverse events reported patient well tolerate",
        "patient well no complaint continue dose",
        "mild nausea report symptom resolve",
        "mild dizziness note transient",
        "severe chest pain refer urgent evaluation",
        "severe dizziness treatment suspend safety",
    ] * 10
    labels = ["No AE", "No AE", "Mild AE", "Mild AE", "Severe AE", "Severe AE"] * 10
    pipeline = build_pipeline(max_features=50)
    pipeline.fit(texts, labels)
    return pipeline, texts


def test_global_importance_returns_ranked_features():
    pipeline, texts = _toy_pipeline()
    explainer = AEShapExplainer(pipeline)
    result = explainer.global_importance(texts, class_label="Severe AE", top_k=5)
    assert len(result["top_features"]) <= 5
    values = [f["mean_abs_shap"] for f in result["top_features"]]
    assert values == sorted(values, reverse=True)


def test_local_explanation_narrative_mentions_a_word_from_note():
    pipeline, texts = _toy_pipeline()
    explainer = AEShapExplainer(pipeline)
    note = "severe chest pain refer urgent evaluation"
    narrative = explainer.business_narrative(note, class_label="Severe AE")
    assert "Severe AE" in narrative
    assert any(word in narrative for word in note.split())
