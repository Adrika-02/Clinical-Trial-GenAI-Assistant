# Architecture

```mermaid
flowchart TD
    subgraph Client["Browser"]
        UI[Streamlit 6-Page App]
    end

    subgraph Interaction["Three Interaction Modes"]
        Chat[Page 2: Chat with Trial Data]
        Form[Page 3: Cohort Explorer]
        Upload[Page 5: Upload and Integrate]
    end

    UI --> Chat
    UI --> Form
    UI --> Upload

    subgraph Agents["LangChain Agent Layer"]
        A1[Agent 1: Data Analysis Agent<br/>NL-to-SQL]
        A2[Agent 2: Clinical Notes Agent<br/>NER + Retrieval]
        A3[Agent 3: Insight Generation Agent<br/>Executive Synthesis]
    end

    Chat --> A1
    Chat --> A2
    Chat --> A3

    subgraph Bedrock["AWS Bedrock"]
        Claude[Claude Sonnet LLM]
    end

    A1 --> Claude
    A2 --> Claude
    A3 --> Claude

    subgraph DataLayer["Data Layer"]
        SQLite[(SQLite:<br/>patients, visits,<br/>adverse_events,<br/>clinical_notes)]
        Pandas[Pandas / NumPy<br/>Data Manipulation]
    end

    A1 -->|generated SQL| SQLite
    SQLite --> Pandas
    Form --> Pandas
    Upload -->|validated CSV/notes| SQLite

    subgraph StatsLayer["Statistical Modelling"]
        Stats[Welch t-test, Chi-square,<br/>Cohen's d, Kruskal-Wallis,<br/>95% CI]
    end

    Pandas --> Stats
    Stats --> UI

    subgraph NLPLayer["NLP / NLU Pipeline"]
        SpaCy[SpaCy: tokenize,<br/>lemmatize, medical NER]
        TFIDF[TF-IDF Vectorizer]
        LogReg[Logistic Regression<br/>AE Classifier]
    end

    Upload --> SpaCy
    A2 --> SpaCy
    SpaCy --> TFIDF --> LogReg
    LogReg --> SQLite

    subgraph Explain["Explainability"]
        SHAP[SHAP: global summary +<br/>local waterfall plots]
    end

    LogReg --> SHAP
    SHAP --> UI

    subgraph MLLayer["Unsupervised ML"]
        KMeans[StandardScaler + K-Means<br/>+ PCA 2D projection]
    end

    Pandas --> KMeans
    KMeans --> UI

    subgraph Reporting["Executive Reporting"]
        PDF[ReportLab PDF Generator]
    end

    Stats --> PDF
    SHAP --> PDF
    KMeans --> PDF
    A3 --> PDF
    PDF --> UI

    subgraph Viz["Visualization"]
        Plotly[Plotly / Seaborn / Matplotlib]
    end

    Pandas --> Plotly
    Stats --> Plotly
    KMeans --> Plotly
    Plotly --> UI
```

## Data flow summary

1. **Browser → Streamlit → LangChain agents → AWS Bedrock**: natural-language questions are routed to one of three agents, which call Claude Sonnet on Bedrock for reasoning/summarization/generation.
2. **Browser → SpaCy NLP → Scikit-learn classifier → SHAP**: clinical notes (typed, searched, or uploaded) are preprocessed, classified for adverse-event severity, and explained token-by-token.
3. **Browser → SQLite → Pandas → Statistical tests → Plotly**: structured cohort queries pull from SQLite, get manipulated in Pandas, are tested for significance, and rendered as charts.

## Component-to-Tredence-skill mapping

| Component | Skill demonstrated |
|---|---|
| Data generation, cohort filtering | Pandas, NumPy |
| AE classifier, K-Means clustering | Scikit-learn (classification + clustering) |
| Welch's t-test, chi-square, Cohen's d, Kruskal-Wallis | Statistical modelling / hypothesis testing |
| SpaCy NER, TF-IDF | NLP / NLU |
| 3 LangChain agents + AWS Bedrock | GenAI, agent-based systems |
| SHAP global/local plots | Interpretable models |
| SQLite schema + aggregation queries | Advanced SQL |
| Plotly/Seaborn/Matplotlib dashboards | Data visualization |
| Business impact metrics throughout | Business audience framing |
