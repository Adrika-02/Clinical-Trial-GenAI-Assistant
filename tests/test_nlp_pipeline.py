from src.nlp.ner import MedicalNER, severity_score
from src.nlp.preprocessing import preprocess_text
from src.nlp.ae_classifier import build_pipeline, CLASSES


def test_preprocess_removes_stopwords_and_lemmatizes():
    cleaned = preprocess_text("The patients were reporting dizziness and were not doing well.")
    assert "the" not in cleaned.split()
    assert "were" not in cleaned.split()
    assert "not" in cleaned.split()  # negation retained


def test_ner_extracts_symptom_severity_and_vitals():
    ner = MedicalNER()
    text = "Moderate dizziness reported. Dose reduced from 20mg to 10mg. BP measured at 138/88. HbA1c improved from 8.2 to 7.1."
    result = ner.extract(text)
    assert "dizziness" in result["symptoms"]
    assert "moderate" in result["severity_descriptors"]
    assert {"systolic": 138, "diastolic": 88} in result["blood_pressure"]
    assert any(v["test"] == "HbA1c" for v in result["lab_values"])
    assert 20 in result["dosages_mg"] and 10 in result["dosages_mg"]


def test_severity_score_scale():
    ner = MedicalNER()
    mild = ner.extract("Patient reports mild nausea.")
    severe = ner.extract("Severe chest pain reported.")
    none = ner.extract("No adverse events reported this cycle.")
    assert severity_score(severe) == 3
    assert severity_score(mild) == 1
    assert severity_score(none) == 0


def test_classifier_pipeline_fits_and_predicts_valid_classes():
    texts = [
        "no adverse events reported patient well",
        "mild nausea reported symptom resolved",
        "severe chest pain patient referred urgent evaluation",
    ] * 20
    labels = ["No AE", "Mild AE", "Severe AE"] * 20
    pipeline = build_pipeline(max_features=100)
    pipeline.fit(texts, labels)
    preds = pipeline.predict(texts[:3])
    assert all(p in CLASSES for p in preds)
