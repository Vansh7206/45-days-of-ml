# Day 5 – Model Evaluation, Improvement & Saving (ML Pipeline)

## 1. Why Model Evaluation is Important
Model evaluation helps us understand:
- How well the model is performing
- Whether the model is overfitting or underfitting
- If the model can generalize to unseen data

Training accuracy alone is NOT enough.

---

## 2. Common Evaluation Metrics (Classification)

### 2.1 Accuracy
- Percentage of correct predictions
- Works well when classes are balanced

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
