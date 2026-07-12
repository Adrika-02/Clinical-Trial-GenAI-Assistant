"""Medical named-entity extraction over clinical notes.

General-purpose spaCy models (en_core_web_sm) are not trained on clinical
vocabulary, so a domain-specific medical NER layer is built on top using
spaCy's rule-based Matcher/PhraseMatcher (symptoms, severity descriptors,
clinical actions) plus targeted regex for numeric clinical values (lab
results, blood pressure readings, drug dosages). This is a standard,
explainable pattern for bootstrapping clinical NER without a large external
medical model (e.g. scispacy), and keeps every extraction traceable to an
exact rule.
"""
import re

from spacy.matcher import PhraseMatcher

from src.nlp.preprocessing import get_nlp

SYMPTOMS = [
    "nausea", "queasiness", "an upset stomach", "feeling sick to the stomach",
    "dizziness", "light-headedness", "feeling unsteady on their feet", "a spinning sensation",
    "elevated blood pressure", "blood pressure trending higher than baseline", "a rise in blood pressure readings",
    "chest pain", "chest discomfort", "tightness in the chest", "pressure in the chest",
    "fatigue", "tiredness", "low energy", "a general sense of exhaustion",
]
SEVERITY_TERMS = ["mild", "moderate", "severe", "significant", "transient"]
ACTION_TERMS = [
    "dose reduced", "dose adjusted", "treatment suspended", "treatment temporarily suspended",
    "discontinuation", "referred for further evaluation", "follow-up scheduled",
    "monitoring closely", "urgent clinical review", "cardiology consult",
]

HBA1C_PATTERN = re.compile(r"HbA1c[^0-9]{0,20}?(\d+\.\d+)(?:\s*(?:to|,)\s*(\d+\.\d+))?", re.IGNORECASE)
BP_PATTERN = re.compile(r"\b(\d{2,3})/(\d{2,3})\b")
DOSE_PATTERN = re.compile(r"\b(\d{1,3})\s?mg\b", re.IGNORECASE)


class MedicalNER:
    def __init__(self, nlp=None):
        self.nlp = nlp or get_nlp()
        self.symptom_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.symptom_matcher.add("SYMPTOM", [self.nlp.make_doc(s) for s in SYMPTOMS])

        self.severity_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.severity_matcher.add("SEVERITY", [self.nlp.make_doc(s) for s in SEVERITY_TERMS])

        self.action_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.action_matcher.add("ACTION", [self.nlp.make_doc(a) for a in ACTION_TERMS])

    def extract(self, text: str) -> dict:
        doc = self.nlp(text)

        symptoms = sorted({doc[s:e].text.lower() for _, s, e in self.symptom_matcher(doc)})
        severity_descriptors = sorted({doc[s:e].text.lower() for _, s, e in self.severity_matcher(doc)})
        actions = sorted({doc[s:e].text.lower() for _, s, e in self.action_matcher(doc)})

        lab_values = []
        for match in HBA1C_PATTERN.finditer(text):
            for group in match.groups():
                if group:
                    lab_values.append({"test": "HbA1c", "value": float(group)})

        blood_pressure = [
            {"systolic": int(m.group(1)), "diastolic": int(m.group(2))} for m in BP_PATTERN.finditer(text)
        ]

        dosages_mg = [int(m.group(1)) for m in DOSE_PATTERN.finditer(text)]

        return {
            "symptoms": symptoms,
            "severity_descriptors": severity_descriptors,
            "clinical_actions": actions,
            "lab_values": lab_values,
            "blood_pressure": blood_pressure,
            "dosages_mg": dosages_mg,
        }


SEVERITY_SCORE_MAP = {"mild": 1, "transient": 1, "moderate": 2, "significant": 2, "severe": 3}


def severity_score(entities: dict) -> int:
    """0-3 scale: highest severity descriptor found, else driven by presence
    of a symptom with no descriptor (treated as mild), else 0."""
    descriptors = entities["severity_descriptors"]
    if descriptors:
        return max(SEVERITY_SCORE_MAP.get(d, 0) for d in descriptors)
    if entities["symptoms"]:
        return 1
    return 0


if __name__ == "__main__":
    ner = MedicalNER()
    sample = (
        "Moderate dizziness reported. Dose reduced from 20mg to 10mg. "
        "BP measured at 138/88. HbA1c improved from 8.2 to 7.1. Follow-up scheduled in 2 weeks."
    )
    result = ner.extract(sample)
    print(result)
    print("Severity score:", severity_score(result))
