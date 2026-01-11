# Day 4 – Logistic Regression (Classification Model)

## 1. What is Logistic Regression?
Logistic Regression is a **supervised machine learning algorithm** used for:
- Binary classification problems
- Output values: **0 or 1**, **Yes or No**, **True or False**

Example use cases:
- Loan Approval (Yes / No)
- Spam Detection
- Disease Prediction

---

## 2. Why Not Linear Regression?
Linear Regression outputs continuous values.
Logistic Regression outputs probabilities between **0 and 1**.

To achieve this, Logistic Regression uses the **Sigmoid Function**.

---

## 3. Sigmoid Function
The sigmoid function converts any value into a range of **0 to 1**.

Formula:
σ(z) = 1 / (1 + e^-z)
- If probability ≥ 0.5 → Class 1
- If probability < 0.5 → Class 0

---

## 4. Logistic Regression Equation
z = w1x1 + w2x2 + ... + b
Then apply sigmoid:
ŷ = σ(z)
---

## 5. Training Logistic Regression in Python

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

predict() → final class (0 or 1)
predict_proba() → probability scores

7. Advantages of Logistic Regression
- Simple and fast
- Easy to interpret
- Works well for linearly separable data

8. Limitations
- Cannot handle complex nonlinear relationships
- Sensitive to outliers
- Requires feature scaling

9. When to Use Logistic Regression
- Binary classification
- Baseline model
- Small to medium datasets

10. Key Takeaways
- Logistic Regression is for classification, not regression
- Uses sigmoid to output probabilities
- Threshold decides final class

