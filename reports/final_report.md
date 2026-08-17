# Final Project Report — Flipkart Customer Service Satisfaction

## 1. Executive Summary

This project develops a customer-service satisfaction classification system for an e-commerce support environment. The solution combines structured customer-service data with optional text feedback and provides a path from data preparation to model deployment and GenAI-assisted interaction summarization.

The supplied project brief defines satisfaction prediction, customer-retention support, agent-performance analysis and conversational summarization as major business use cases.

## 2. Problem Statement

The objective is to identify whether a customer is satisfied or unsatisfied after a support interaction and identify the operational factors associated with dissatisfaction.

## 3. Business Objectives

- Detect dissatisfied customers early.
- Reduce manual QA review.
- Identify service bottlenecks.
- Support agent coaching.
- Improve customer retention.
- Summarize long support interactions.

## 4. Data

The brief specifies a CSV dataset containing customer interaction and satisfaction information. The actual CSV was not included with the supplied project brief, so dataset-specific counts and metrics must be generated after the CSV is added.

## 5. Data Preparation

Pipeline:
1. Load CSV.
2. Remove duplicates.
3. Remove rows without target.
4. Trim string fields.
5. Detect date/time fields.
6. Create year/month/day/hour/day-of-week features.
7. Create text-length and word-count features.
8. Impute numeric and categorical missing values.
9. One-hot encode categorical features.
10. Apply TF-IDF to text feedback where available.

## 6. Target

For numeric CSAT-style 1–5 scores, the default binary business framing is:
- 1–3: Unsatisfied
- 4–5: Satisfied

This rule is implemented in the pipeline and can be changed in `src/data_utils.py` if the project owner requires a different definition.

## 7. Modeling

The pipeline compares:
- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting

Evaluation:
- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1
- ROC-AUC for binary classification
- Confusion matrix

## 8. Explainability

Tree-based feature importance is exported when supported by the selected model. SHAP can be added for deeper local/global explanations.

## 9. GenAI

Azure OpenAI is used as an optional layer for:
- Interaction summarization
- Sentiment/context extraction
- Recommended action generation

The GenAI output should support supervisors rather than replace human decisions.

## 10. Deployment

The model can be deployed as an Azure ML online endpoint and consumed through a REST API or support application.

## 11. Dashboard

Power BI should show:
- Total interactions
- Satisfaction rate
- Unsatisfied interaction count
- Satisfaction by channel
- Satisfaction by issue category
- Agent performance
- Response/resolution time
- Monthly trend
- Top dissatisfaction drivers

## 12. Results

Run `python run_project.py` after adding the actual CSV. The generated `reports/model_metrics.csv` contains the actual model results.

The original project brief lists example outcome figures, but they are not treated as verified results here because the source dataset was not supplied.

## 13. Recommendations

Use the model to prioritize high-risk interactions, investigate long response/resolution times, identify recurring complaint categories, and target coaching toward operational factors associated with low satisfaction.

## 14. Limitations

- Results depend on the supplied dataset.
- Historical labels may contain survey bias.
- Text feedback can contain sensitive information.
- Prediction performance may drift as customer behavior and support processes change.
- GenAI outputs require validation and responsible-AI controls.
