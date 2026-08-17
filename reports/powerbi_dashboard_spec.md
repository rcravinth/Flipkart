# Power BI Dashboard Specification

## Page 1 — Executive Overview

Cards:
- Total Interactions
- Satisfied %
- Unsatisfied %
- Average predicted satisfaction probability
- Average response time
- Average resolution time

Charts:
- Satisfaction trend by month
- Satisfaction by channel
- Satisfaction by product category

## Page 2 — Agent Performance

- Agent-wise interaction count
- Agent-wise satisfaction rate
- Average response time
- Average resolution time
- Unsatisfied cases

Use slicers for:
- Date
- Channel
- Product category
- Issue type
- Agent

## Page 3 — Dissatisfaction Drivers

- Top issue categories
- Top complaint keywords
- Response-time buckets
- Resolution-time buckets
- Product category vs satisfaction

## Page 4 — Prediction Monitoring

- Actual vs predicted satisfaction
- Confidence/probability distribution
- False positive / false negative counts
- Model performance over time

## Suggested DAX

Satisfied Rate =
DIVIDE(
    CALCULATE(COUNTROWS(Predictions), Predictions[prediction] = 1),
    COUNTROWS(Predictions)
)

Unsatisfied Count =
CALCULATE(
    COUNTROWS(Predictions),
    Predictions[prediction] = 0
)
