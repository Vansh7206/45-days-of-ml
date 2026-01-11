# Day 2 – Data Understanding (Human View vs Model View)

## What a dataset represents
A dataset is a collection of historical snapshots.
Each row represents information captured at a specific moment in time.
It does not represent full reality, future behavior, or hidden intent.

Machine learning models assume that these snapshots contain enough information
to predict an outcome, which is a strong and often incorrect assumption.

---

## Human View vs Model View of Features
## Columns are mentioned below:-

### ApplicantIncome
Human view:
- Indicates salary and financial stability
- High Salary = Better Option 

Model view:
- A numeric value with a distribution

Risk:
- The model blindly trusts reported income and cannot judge stability or authenticity

---

### CoapplicantIncome
Human view:
- Additional household income and support

Model view:
- Often treated as zero vs non-zero

Risk:
- The model may overvalue the presence of a coapplicant without understanding dependency

---

### LoanAmount
Human view:
- Higher loan amount means higher risk

Model view:
- A continuous numeric feature

Risk:
- Loan amount is often derived from income, leading to double counting of information

---

### Loan_Amount_Term
Human view:
- Longer term reduces EMI burden

Model view:
- Numeric column with low variation

Risk:
- Low variance reduces usefulness and may add noise

---

### Gender
Human view:
- Demographic information

Model view:
- Encoded binary or categorical variable

Risk:
- The model may learn historical or social bias instead of financial risk

---

### Married
Human view:
- Indicates household stability

Model view:
- Binary indicator

Risk:
- Correlation may reflect bias rather than repayment ability

---

### Education
Human view:
- Higher education implies better job prospects

Model view:
- Encoded categorical value

Risk:
- Past approval patterns may encode bias instead of true signal

---

### Self_Employed
Human view:
- Income instability risk

Model view:
- Binary feature

Risk:
- Income volatility is not captured, leading to oversimplification

---

### Property_Area
Human view:
- Location-based opportunity and income potential

Model view:
- Encoded category without geographic meaning

Risk:
- False ordering and policy bias may influence predictions

---

### Credit_History
Human view:
- Past repayment behavior and trustworthiness

Model view:
- Strong binary predictor

Risk:
- This feature may encode the bank’s decision rule itself,
  causing the model to learn policy instead of real-world risk.
  High accuracy from this feature can be misleading.

---

### Loan_Status (Target)
Human view:
- Correct or incorrect loan decision

Model view:
- Label used to minimize prediction error

Reality:
- The model learns historical bank decisions, not actual repayment outcomes

---

## Why powerful features can be dangerous

Features that produce very high accuracy may:
- Dominate model learning
- Hide weaker but meaningful signals
- Encode human or policy bias

High accuracy does not guarantee a good or reliable model.

---

## Key Takeaways
- Models see numbers, not meaning
- Correlation is not causation
- Strong features should be questioned, not trusted
- Data understanding is required before any modeling
