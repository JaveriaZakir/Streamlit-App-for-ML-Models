import streamlit as st
import pandas as pd
from joblib import load
import dill

#LOAD MODEL FEATURES
with open('pipeline.pkl', 'rb') as f:
    model=dill.load(f)

my_feature_dict=load('my_feature_dict.pkl')

#PREDICTION LOGIC
def predict_churn(input_df):
    return model.predict(input_df)

#APPLICATION USER INTERFACE
st.title("Employee Churn Predictor")
st.subheader("Based on Employee Dataset")
# CATEGORICAL FEATURES:
st.subheader("Categorical Details")
categorical_info = my_feature_dict.get('CATEGORICAL')
categorical_data = {}
for idx, feature in enumerate(categorical_info.get('Column Name').values()):
    options = categorical_info.get('Members')[idx]
    categorical_data[feature] = st.selectbox(f"{feature}", options, key=f"cat_{feature}")

# NUMERICAL FEATURES:
st.subheader("Job & Personal Information")
numerical_info = my_feature_dict.get('NUMERICAL')
numerical_data = {}

for feature in numerical_info.get('Column Name'):
    if feature == "Age":
        numerical_data[feature] = st.slider("Age", 18,70,18, key=f"num_{feature}")
    elif feature == "JoiningYear":
        numerical_data[feature] = st.number_input("Year of Joining", min_value=1992, max_value=2025, value=2018, key=f"num_{feature}")
    elif feature == "PaymentTier":
        numerical_data[feature] = st.radio("Payment Tier",[1, 2, 3],index=1, key=f"num_{feature}")
    elif feature == "ExperienceInCurrentDomain":
        numerical_data[feature] = st.slider("Experience (Years)",0,30,1, key=f"num_{feature}")
    else:
        numerical_data[feature] = st.number_input(f"{feature}",key=f"num_{feature}")
combined_input = {**categorical_data, **numerical_data}
input_df = pd.DataFrame([combined_input])
#PREDICTION
if st.button("Predict"):
    result = predict_churn(input_df)[0]
    meaning = {"Yes": "expected", "No": "not expected"}.get(result, "unknown")
    st.markdown(f"The model predicts the employee is *{meaning}* to churn.")
    
#CREATED BY
st.markdown("---")
st.caption("Created by Javeria Zakir")
