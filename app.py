import streamlit as st
import joblib
import numpy as np

model  = joblib.load('bankruptcy_best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('🏦 Bankruptcy Prediction App')
st.write('Enter company financial details below:')

company_id          = st.number_input('Company ID (encoded)', value=0)
industry            = st.number_input('Industry (encoded)', value=0)
years_in_operation  = st.number_input('Years in Operation', value=10)
annual_revenue      = st.number_input('Annual Revenue', value=500000)
net_profit          = st.number_input('Net Profit', value=50000)
total_assets        = st.number_input('Total Assets', value=300000)
total_liabilities   = st.number_input('Total Liabilities', value=150000)
debt_to_equity      = st.number_input('Debt to Equity Ratio', value=1.5)
current_ratio       = st.number_input('Current Ratio', value=2.0)
cash_flow           = st.number_input('Cash Flow', value=30000)
credit_rating       = st.number_input('Credit Rating (encoded)', value=2)
employee_count      = st.number_input('Employee Count', value=50)
market_trend        = st.number_input('Market Trend (encoded)', value=1)

if st.button('Predict'):
    features = np.array([[company_id, industry, years_in_operation,
                          annual_revenue, net_profit, total_assets,
                          total_liabilities, debt_to_equity, current_ratio,
                          cash_flow, credit_rating, employee_count, market_trend]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error(f'⚠️ Bankrupt — {probability*100:.1f}% probability')
    else:
        st.success(f'✅ Not Bankrupt — {probability*100:.1f}% bankruptcy risk')