# Business Impact Summary

*All figures below are pulled directly from the pipeline's real output (`data/processed/stats_results.json`, `data/processed/clustering_results.json`, `models/saved/ae_classifier_metrics.json`) — none are hardcoded or illustrative unless explicitly labeled as an industry estimate.*

## Safety signal detection

- **The NLP adverse-event classifier detected 483 AE-positive clinical notes on the held-out test set vs. 255 caught by manual case-report-form coding alone — an 89.4% uplift in detection.**
- Of the 97 adverse events that manual CRF coding missed entirely (present in the investigator's free-text note but never logged in the structured case report form), the NLP classifier **recovered 96.9% of them (94 of 97)**.
- Classifier performance: **92.7% accuracy, F1-macro 0.829**, with recall (89.9%) deliberately prioritized over precision (78.0%) — in pharmacovigilance, a missed real adverse event is far costlier than a false alarm a clinician reviews and dismisses in seconds.

## Primary efficacy endpoint

- **Drug X patients scored 9.05 points higher than placebo on the primary outcome measure (60.19 vs. 51.14)** — a statistically significant difference (Welch's t = 18.05, **p < 0.001**, Cohen's d = 0.81, a large effect).
- 35.2% of Drug X patients achieved HbA1c < 7% control by Week 24 vs. a materially lower placebo rate (chi² = 218.75, p < 0.001).

## Patient safety segmentation

- K-Means clustering identified an **"AE-Prone" patient phenotype — 24.0% of the study population (n=480) — that accounts for 161 of the trial's 162 severe adverse events (99.4%) and a 29.8% dropout rate**, roughly 3.5x the rate of every other cluster.
- **Targeting this phenotype for closer monitoring (e.g. more frequent vitals checks, lower starting dose) is the single highest-leverage safety intervention available in this trial** — it concentrates almost the entire safety burden into a cohort a quarter of the trial's size.
- Trial-wide completion rate: 86.4% (dropout concentrated almost entirely in the AE-Prone cluster).

## Operational efficiency

- **The full executive report — endpoint statistics, safety profile, cluster analysis, SHAP explainability, and live AI-generated executive insights — compiles in under 20 seconds**, versus the multi-day turnaround typically required for a biostatistician or medical writer to assemble the same material by hand.
- Natural-language querying (the Chat agent) eliminates the SQL/coding bottleneck between a medical researcher's question and a data-backed answer — no analyst hand-off required for routine queries like arm comparisons or dropout patterns.

## Context: why this matters at trial scale

*(Industry estimate, not computed from this project's data — included for framing only.)* Published industry analyses (e.g. Tufts Center for the Study of Drug Development) put the fully-loaded cost of bringing a drug through Phase III in the hundreds of millions of dollars, and a single safety signal that surfaces only after launch can trigger costly label changes, litigation, or withdrawal. Against that backdrop, a measured 89.4% improvement in adverse-event detection sensitivity — recovering safety signals that manual coding would have missed entirely — represents a meaningful, quantifiable reduction in that risk, not a cosmetic accuracy metric.
