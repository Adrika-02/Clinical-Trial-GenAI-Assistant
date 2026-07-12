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
- [x] Step 2 — Synthetic data generation + SQLite (500 patients, 3,000 visits, 3,000 AE records, 3,000 clinical notes)
- [x] Step 3 — Statistical analysis module (Welch's t-test, chi-square, Cohen's d, Kruskal-Wallis, 95% CIs)
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

Generate the synthetic clinical trial dataset and SQLite database:

```bash
python -m src.data_generation.build_database
```

This produces `data/clinical_trial.db` with four tables (`patients`, `visits`, `adverse_events`, `clinical_notes`) plus raw CSV exports in `data/raw/` for use as upload-demo files. Current output: **500 patients** (250 Drug X / 250 Placebo), **3,000 visits** (6 per patient), **3,000 adverse-event records**, **3,000 clinical notes**. The treatment effect and adverse-event burden are simulated (not hardcoded) via a per-patient dose-response curve, so every downstream statistical test and ML model is fit against a genuinely emergent signal — e.g. mean HbA1c drops from 8.11→7.11 in the Drug X arm vs 8.20→7.94 on placebo by Week 24, and Drug X carries a higher adverse-event and dropout rate, consistent with an active drug vs. placebo comparison.

Run the statistical analysis suite (writes `data/processed/stats_results.json`):

```bash
python -m src.stats.run_analysis
```

## Statistical results (real computed output, Week 24 primary endpoint)

| Test | Result |
|---|---|
| Welch's t-test (primary outcome score, Drug X vs Placebo) | t=7.98, **p<0.001**, means 60.05 vs 52.15 |
| Cohen's d (effect size) | **d=0.71** (medium-to-large effect) |
| Chi-square (HbA1c <7% control at Week 24) | chi2=56.11, dof=1, **p<0.001** |
| Chi-square (adverse event occurrence, any severity) | chi2=43.35, dof=1, **p<0.001** |
| Chi-square (AE severity class distribution) | chi2=49.90, dof=2, **p<0.001** |
| Kruskal-Wallis (HbA1c reduction by worst AE severity) | H=35.41, **p<0.001** |

All values are computed live from the generated dataset via `scipy.stats` / `statsmodels` — none are hardcoded. Re-running `build_database` with a different seed will change these numbers, which is the point: the pipeline recomputes real statistics against whatever data currently lives in SQLite. Unit tests for the stats module are in `tests/test_statistical_tests.py` (run with `python -m pytest`).

Further pipeline and app run instructions will be added as each step lands.

## License

MIT
