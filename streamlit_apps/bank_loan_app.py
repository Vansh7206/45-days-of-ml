import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#page configuration
st.set_page_config(page_title='Bank loan risk model',page_icon='🏦',layout='centered')
st.title('🏦 Bank Loan Default Risk Model')
st.write('A model made for Educational Purpose')

#Calculated threshold
THRESHOLD = 0.3

#Data loading
@st.cache_data
def load_data():
    return pd.read_csv(r'C:\Users\vchan\OneDrive\Desktop\45-days-of-ml\datasets\bank_loan_data.csv')

df = load_data()

#assigning X and y values
X = df.drop('default',axis=1)
y = df['default']

#Testing data
X_temp,X_test,y_temp,y_test = train_test_split(X,y,random_state=42,test_size=0.3,stratify=y)

#Training and Validating Data
X_train,X_val,y_train,y_val = train_test_split(X_temp,y_temp,random_state=42,test_size=0.2,stratify=y_temp)

#Log1p on values
for col in ['income','loan_amount']:
    X_train[col] = np.log1p(X_train[col])
    X_val[col] = np.log1p(X_val[col])
    X_test[col] = np.log1p(X_test[col])

#Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#model building
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled,y_train)

#Probablity on test data
y_test_proba = model.predict_proba(X_test_scaled)[:,1]
y_test_pred = (y_test_proba > THRESHOLD).astype(int)

#confusion matrix
cm = confusion_matrix(y_test,y_test_pred)
tn,fp,fn,tp = cm.ravel()

#Page design and Dataset
st.subheader("🧾 Dataset Preview")
st.dataframe(df, use_container_width=True)

#Confusin matrix design
st.write('### 📊 Confusion Matrix of the trained Data')
st.write(pd.DataFrame(cm,index=['Actual Good','Actual Bad'], columns=['Predicted Good','Predicted Bad']))

#Buisness info
st.info(
    f"""
**Business Interpretation**
- FN (bad loans approved) = {fn}  ← highest risk
- Threshold fixed at **{THRESHOLD}**
- Model trades business for safety (bank-first logic)
"""
)

# UserInput
st.subheader('🔍 Check for New Customer')

age = st.slider('Age',21,65,25)
income = st.number_input('Annual Income',20000,150000,30000)
credit_score = st.slider('Credit Score',300,850,400)
loan_amt = st.number_input('Loan Amount',5000,500000,100000)
existing_loan = st.slider('Existing Loans',0,5,2)

if st.button('🔮 Evaluate Customer'):
    user_input = pd.DataFrame({
    'age': [age],
    'income': [np.log1p(income)],
    'credit_score': [credit_score],
    'loan_amount': [np.log1p(loan_amt)],
    'existing_loans': [existing_loan]
})


    user_input_scaled = scaler.transform(user_input)
    user_probablity = model.predict_proba(user_input_scaled)[0,1]
    prediction = int(user_probablity > THRESHOLD)

    st.write(f"### Probability of Default: **{user_probablity:.2f}**")

    if prediction == 1:
        st.error("❌ Loan Rejected (High Risk)")
    else:
        st.success("✅ Loan Approved (Acceptable Risk)")
