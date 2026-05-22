# Prediction Explanation

Model: **Logistic Regression**
Customer ID: **9497-QCMMS**
Row index: **1976**
Predicted label: **Yes**
Predicted churn probability: **93.90%**

Positive SHAP values increase churn risk. Negative SHAP values decrease churn risk.

## Customer Context

| Field | Value |
| --- | --- |
| customerID | 9497-QCMMS |
| Contract | Month-to-month |
| tenure | 1 |
| MonthlyCharges | 93.55 |
| TotalCharges | 93.55 |
| InternetService | Fiber optic |
| PaymentMethod | Electronic check |
| TechSupport | No |
| OnlineSecurity | No |
| Churn | Yes |

## Top Feature Contributions

| Feature | Direction | SHAP value | Transformed value |
| --- | --- | ---: | ---: |
| tenure | increases churn risk | 1.5530 | -1.2816 |
| MonthlyCharges | decreases churn risk | -0.6395 | 0.9497 |
| TotalCharges | decreases churn risk | -0.4678 | -0.9693 |
| InternetService_Fiber optic | increases churn risk | 0.3697 | 1.0000 |
| Contract_Month-to-month | increases churn risk | 0.3036 | 1.0000 |
| Contract_Two year | increases churn risk | 0.2018 | 0.0000 |
| InternetService_DSL | increases churn risk | 0.1868 | 0.0000 |
| StreamingMovies_Yes | increases churn risk | 0.1863 | 1.0000 |
| PaymentMethod_Electronic check | increases churn risk | 0.1734 | 1.0000 |
| StreamingTV_Yes | increases churn risk | 0.1536 | 1.0000 |

## Model Output Note

For linear and tree explainers, SHAP values explain model output space; probability is reported separately.
