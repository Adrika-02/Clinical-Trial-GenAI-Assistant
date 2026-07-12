# Resume Bullets

*Every number below is pulled directly from real pipeline output — see [business_impact.md](business_impact.md) and the README's per-step "real output" sections for the underlying source.*

## Bullet 1 — Life Sciences domain, NLP, statistics

> Built an end-to-end clinical trial analytics pipeline over 2,000 Phase III patients and 12,000 unstructured clinical notes, engineering an NLP adverse-event classifier (SpaCy preprocessing/NER, TF-IDF, Logistic Regression) that achieved 92.7% accuracy (F1=0.83) and recovered 96.9% of adverse events missed by manual case-report-form coding, with SHAP-based global and local explainability and statistical validation (Welch's t-test, chi-square, Cohen's d, Kruskal-Wallis, all p<0.001).

## Bullet 2 — GenAI agents, clustering, dashboard, deployment

> Architected a provider-agnostic 3-agent GenAI system (LangChain; swappable across AWS Bedrock/Claude, Anthropic, and Groq behind a single factory function) for natural-language SQL querying, clinical-note summarization, and executive insight generation over a SQLite data warehouse, paired with K-Means patient segmentation that isolated a high-risk cluster (24% of patients) responsible for 99.4% of severe adverse events; shipped as a 6-page Streamlit dashboard with on-demand PDF reporting, deployed live on Streamlit Cloud.

## Shorter variants (if character-limited)

**Bullet 1 (short):** NLP adverse-event classifier (SpaCy + TF-IDF + Logistic Regression) over 2,000 patients / 12,000 clinical notes — 92.7% accuracy, F1=0.83, SHAP-explainable, 96.9% recovery of manually-missed AEs; validated efficacy via Welch's t-test, chi-square, Cohen's d (p<0.001).

**Bullet 2 (short):** 3-agent LangChain GenAI system (provider-agnostic across AWS Bedrock/Claude, Anthropic, and Groq) for NL-to-SQL, note summarization, and executive insights, combined with K-Means clustering (isolated an AE-prone cohort behind 99.4% of severe events) in a deployed 6-page Streamlit dashboard with auto-generated PDF reporting.
