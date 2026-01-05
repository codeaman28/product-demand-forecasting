import streamlit as st
import pandas as pd
import joblib

st.write("Streamlit app started")  # debug line

model = joblib.load("model/demand_model.pkl")

st.title("Product Demand Forecasting System")

store = st.number_input("Store ID", min_value=1)
item = st.number_input("Item ID", min_value=1)
month = st.slider("Month", 1, 12)
day = st.slider("Day", 1, 31)

if st.button("Predict Demand"):
    input_data = pd.DataFrame(
        [[store, item, month, day]],
        columns=["store", "item", "month", "day"]
    )
    prediction = model.predict(input_data)
    st.success(f"Predicted Sales: {int(prediction[0])} units")
