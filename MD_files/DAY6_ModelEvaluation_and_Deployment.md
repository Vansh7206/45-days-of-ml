Day 6 – Model Evaluation + Deployment (End-to-End ML Pipeline)
1. Why Evaluation Alone Is Not Enough
- A model with good metrics is still useless if:
- It cannot be used by non-technical users
- It cannot handle real input
- It cannot be deployed in production

Evaluation tells you how good the model is.
Deployment proves whether the model is usable.

2. Re-checking Model Performance Before Deployment
- Before deployment, always verify:

### 2.1 Confusion Matrix
Shows:
- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

This helps understand real business impact, not just accuracy.

2.2 Precision, Recall, F1-Score
- Used when accuracy is misleading.

### Precision → How many predicted positives were correct
### Recall → How many actual positives were caught

F1-Score → Balance between precision and recall

3. Threshold Tuning (Critical for Deployment)
By default:
- Logistic Regression uses 0.5 threshold

But in real systems:
- Loan approval
- Fraud detection
- Medical diagnosis

Threshold must be adjusted based on risk.

##### Changing threshold directly affects:
- False approvals
- False rejections

4. Freezing the Final Model
- Consistent predictions
- No data leakage
- Stable deployment

5. Real-World Deployment Mindset
- A deployed ML model must:
- Be interpretable
- Be stable
- Handle bad input
- Reflect business logic
- Fail safely

### Accuracy without usability = zero value.

