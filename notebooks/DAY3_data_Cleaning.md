Day 3 — Data Cleaning & Feature Handling
1. Goal of Day 4
- The goal of Day 4 was to convert raw data into model-ready data.

Key idea:
- Preprocessing is not a mechanical step; it decides what the model is allowed to learn.

2. Missing Value Handling (Column-Wise Decisions)
- Missing values were not treated uniformly.
- Each column was handled based on meaning and risk.    

1. Categorical Columns
- Gender
- Married
- Dependents
- Self_Employed

Strategy:
- Filled with mode

Reason:
- Small number of missing values
- Mode preserves category distribution
- Low risk of distortion

2. LoanAmount

Strategy:
- Filled with median

Reason:
- Highly skewed
- Presence of outliers
- Median is robust, mean would distort distribution

3. Loan_Amount_Term

Strategy:
- Filled with mode

Reason:
- Most loans share the same term (360)
- Behaves almost like a categorical variable

4. Credit_History (Special Case)

Strategy:
- Filled with 0

Reason:
- Missing credit history often means new or thin credit profile
- Missingness itself is informative
- Explicit assumption preferred over hiding uncertainty

# More notes in 'Day3_Data_Cleaning&Feature_Handling.ipynb'
