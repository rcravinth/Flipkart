# Flipkart Customer Service Satisfaction — ML & GenAI Project

## Objective
Build an end-to-end customer-satisfaction classification solution for e-commerce support interactions.

The project follows the supplied project brief:
- Binary/multiclass satisfaction classification
- Data cleaning and EDA
- Feature engineering
- Multiple ML models
- Class-imbalance handling
- Model evaluation and explainability
- Streamlit inference UI
- Azure ML deployment skeleton
- GenAI summarization skeleton
- Power BI-ready prediction export

> **Important:** The supplied brief references a CSV dataset, but the CSV itself was not attached. Therefore, this package does not invent dataset-specific results. Run the pipeline after placing the real CSV in `data/flipkart_customer_service.csv`.

## Project structure

```text
flipkart_customer_satisfaction_project/
├── app/
│   └── streamlit_app.py
├── azure/
│   ├── conda.yml
│   └── deploy_azure_ml.py
├── data/
│   └── README.md
├── models/
├── notebooks/
│   └── 01_end_to_end_project.ipynb
├── reports/
│   ├── final_report.md
│   └── powerbi_dashboard_spec.md
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── eda.py
│   ├── train.py
│   ├── predict.py
│   ├── explain.py
│   └── genai_summary.py
├── .gitignore
├── requirements.txt
└── run_project.py
```

## 1. Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Add the dataset

Copy the supplied CSV to:

```text
data/flipkart_customer_service.csv
```

If your filename is different, change `DATA_PATH` in `src/config.py`.

The code automatically searches for a likely target column using names such as:
`CSAT Score`, `CSAT_Score`, `CSAT`, `satisfaction`, `satisfaction_score`.

If the target cannot be identified automatically, set `TARGET_COLUMN` manually in `src/config.py`.

## 3. Run the complete ML pipeline

```bash
python run_project.py
```

Outputs:
- `reports/eda_summary.json`
- `reports/model_metrics.csv`
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`
- `reports/feature_importance.csv`
- `models/best_model.joblib`
- `models/metadata.json`
- `reports/predictions.csv`

## 4. Run the web application

```bash
streamlit run app/streamlit_app.py
```

The app supports:
- CSV upload
- Satisfaction prediction
- Probability display
- Batch prediction download

## 5. GenAI summarization

The GenAI component is intentionally separated from the core classifier.

Set:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_DEPLOYMENT
AZURE_OPENAI_API_VERSION
```

Then use:

```bash
python -c "from src.genai_summary import summarize_text; print(summarize_text('Customer waited three days and received the wrong product.'))"
```

## 6. Azure ML deployment

Edit the values in `azure/deploy_azure_ml.py`, authenticate with Azure, and run:

```bash
python azure/deploy_azure_ml.py
```

This is a deployment skeleton; actual Azure resource names and credentials must come from the user's Azure subscription.

## 7. Power BI

Import `reports/predictions.csv` into Power BI. Recommended pages and measures are documented in:

`reports/powerbi_dashboard_spec.md`

## 8. Academic/project presentation flow

1. Business problem
2. Dataset and data dictionary
3. Data quality
4. EDA
5. Feature engineering
6. Class balancing
7. Model comparison
8. Hyperparameter tuning
9. Final model
10. Explainability
11. GenAI summarization
12. Deployment architecture
13. Dashboard
14. Business recommendations
15. Limitations and next steps

## Important result policy

The original brief contains example/project-brief results such as 87% accuracy and F1 0.84. Those figures are **not reproduced as actual results by this package**, because the underlying CSV was not provided. The pipeline calculates the real metrics from the supplied dataset.
