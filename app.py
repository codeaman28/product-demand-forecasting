import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.write("Streamlit app started")

# -----------------------------
# TRAIN MODEL INSIDE APP
# -----------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("data/sales.csv")

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    X = df[["store", "item", "month", "day"]]
    y = df["sales"]

    model = LinearRegression()
    model.fit(X, y)

    return model

model = train_model()

# -----------------------------
# UI PART
# -----------------------------
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
