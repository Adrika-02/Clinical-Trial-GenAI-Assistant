# Clinical Trial Intelligence Assistant

*A GenAI-powered analytics platform for clinical trial data — built for the Life Sciences analytics domain.*

> **Status: under active development.** This README is updated after each build step with real metrics (no placeholders). See [docs/architecture.md](docs/architecture.md) for the full system diagram.

## The problem

Clinical research teams sit on two data streams that rarely talk to each other: structured EDC data (labs, vitals, adverse event codes) and unstructured clinical notes written by investigators at every visit. Safety signals — the early warning signs of an adverse drug reaction — are frequently buried in free text that never gets systematically mined, and cohort-level statistical comparisons typically require a biostatistician and a multi-day turnaround. In a Phase III trial, a single missed safety signal can cost millions in downstream liability and delay.

This project builds an assistant that lets a medical researcher ask plain-English questions, filter cohorts through a form, or upload new data — and get statistically rigorous, explainable answers back in seconds, with an auto-generated executive PDF at the end.

## Three ways to interact

1. **Chat** — ask any question in natural language; a LangChain agent converts it to SQL, queries SQLite, and returns a data-backed answer with an auto-generated chart.
2. **Form-based cohort explorer** — filter by age, arm, AE severity, etc.; get instant statistical comparison against the full population.
3. **Upload** — drop in a new patient CSV or a clinical notes file; the pipeline validates it, runs NLP adverse-event classification, and makes it queryable immediately.

## Tech stack

`Python 3.11` · `Pandas` / `NumPy` · `Scikit-learn` · `SciPy` / `statsmodels` · `SpaCy` · `AWS Bedrock (Claude Sonnet)` · `LangChain` · `SQLite` · `SHAP` · `Plotly` / `Seaborn` / `Matplotlib` · `Streamlit` · `ReportLab`

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full Mermaid flowchart and the skill-to-component mapping used for interview prep.

## Project structure

```
clinical-trial-genai-assistant/
├── src/
│   ├── data_generation/   # synthetic patient/visit/notes generator
│   ├── db/                # SQLite schema + access layer
│   ├── stats/             # hypothesis tests, effect sizes, CIs
│   ├── nlp/                # SpaCy preprocessing, NER, TF-IDF + LogReg AE classifier
│   ├── explainability/     # SHAP global + local explanations
│   ├── clustering/         # K-Means + PCA patient segmentation
│   ├── agents/             # 3 LangChain agents (Bedrock-backed)
│   └── reports/            # ReportLab executive PDF generator
├── app/                   # 6-page Streamlit dashboard
├── models/                # saved classifier + SHAP plots (generated, gitignored)
├── data/                  # SQLite DB + generated CSVs (gitignored)
├── docs/                  # architecture diagram, interview prep, screenshots
└── tests/
```

## Status

- [x] Step 1 — Project scaffold + architecture diagram
- [ ] Step 2 — Synthetic data generation + SQLite
- [ ] Step 3 — Statistical analysis module
- [ ] Step 4 — NLP adverse event classifier
- [ ] Step 5 — SHAP explainability
- [ ] Step 6 — K-Means patient clustering
- [ ] Step 7 — LangChain agents (AWS Bedrock)
- [ ] Step 8 — Streamlit dashboard (6 pages)
- [ ] Step 9 — PDF executive report
- [ ] Step 10 — Business impact metrics
- [ ] Deployed live URL

## How to run locally

```bash
conda create -n clinical-trial-ai python=3.11 -y
conda activate clinical-trial-ai
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env   # then fill in your AWS Bedrock credentials
```

Pipeline and app run instructions will be added as each step lands.

## License

MIT
