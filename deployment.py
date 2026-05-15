import streamlit as st
import joblib
import numpy as np

model  = joblib.load('bankruptcy dataset.pkl')
scaler = joblib.load('ss.pkl')

st.title('Bankruptcy Prediction')
st.write('Enter company financial details below:')

company_id          = st.text_input("Company ID", placeholder="e.g. F22781")
industry            = st.selectbox("Industry",options=["Manufacturing", "Retail", "IT", "Construction", "Healthcare", "Finance"])
years_in_operation  = st.slider("Years in Operation", min_value=0, max_value=50, value=0, step=1)
annual_revenue      = st.number_input("Annual Revenue", min_value=0,step=1000)
net_profit          = st.number_input("Net Profit", value=0, step=1000)
total_assets        = st.number_input("Total Assets", min_value=0, step=1000)
total_liabilities   = st.number_input("Total Liabilities", min_value=0, step=1000)
debt_to_equity      = st.slider("Debt to Equity Ratio", min_value=0.0, max_value=20.0, step=0.1)
current_ratio       = st.slider("Current Ratio", min_value=0.0, max_value=10.0, step=0.1)
cash_flow           = st.number_input("Cash Flow",step=1000.0, format="%.2f")
credit_rating       = st.radio("Credit Rating",options=["A", "B", "C", "D"],horizontal=True)
employee_count      = st.number_input("Employee Count", min_value=0, step=1)
market_trend        = st.selectbox("Market Trend",options=["Growing", "Stable", "Declining"])

industry_map = {"Manufacturing": 0,"Retail": 1,"IT": 2,"Construction": 3,"Healthcare": 4,"Finance": 5}
 
credit_rating_map = {"A": 3, "B": 2, "C": 1, "D": 0}
 
market_trend_map = {"Growing": 2, "Stable": 1, "Declining": 0}

final  = joblib.load(r'bankruptcy dataset.pkl')
label1 = joblib.load(r'le.pkl')
label2 = joblib.load(r'le1.pkl')
label3 = joblib.load(r'le2.pkl')
label4 = joblib.load(r'le3.pkl')
stand  = joblib.load(r'ss.pkl')


 
features_scaled = stand.transform(features)
prediction = model.predict(features_scaled)[0]

if st.button("Predict Bankruptcy Risk"):features = np.array([[industry_map[industry],years_in_operation,annual_revenue,net_profit,total_assets,
total_liabilities,debt_to_equity,current_ratio,cash_flow,credit_rating_map[credit_rating],employee_count,market_trend_map[market_trend],]])

st.divider()
 

if prediction == 1:
        st.error("Bankrupt")
else:
        st.success("Not Bankrupt")
