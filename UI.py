import streamlit as st
import numpy as np
import h5py
import joblib
import io

def load_model_from_h5(h5_path):
    with h5py.File(h5_path, "r") as h5f:
        model_bytes = h5f["model"][()].tobytes()
        scaler_bytes = h5f["scaler"][()].tobytes()

    model = joblib.load(io.BytesIO(model_bytes))
    scaler = joblib.load(io.BytesIO(scaler_bytes))
    return model, scaler

# Load model and scaler
model, scaler = load_model_from_h5("logistic_model.h5")

# Streamlit UI
st.title("🛒 Delivery Delay Predictor")
st.write("Enter order details to predict if the delivery will be delayed.")

platform = st.selectbox("Platform", ["Blinkit", "JioMart", "BigBasket"])
product_category = st.selectbox("Product Category", ["Beverages", "Dairy", "Fruits & Vegetables"])
order_value = st.number_input("Order Value (INR)", min_value=0, max_value=2000, value=300)
delivery_time = st.slider("Delivery Time (Minutes)", 5, 120, 30)
service_rating = st.slider("Service Rating (1 = Bad, 5 = Excellent)", 1, 5, 4)
refund_requested = st.selectbox("Refund Requested?", ["No", "Yes"])

platform_map = {"Blinkit": 0, "JioMart": 1, "BigBasket": 2}
category_map = {"Beverages": 0, "Dairy": 1, "Fruits & Vegetables": 2}
refund_map = {"No": 0, "Yes": 1}

if st.button("Predict Delay"):
    input_data = np.array([
        platform_map[platform],
        delivery_time,
        category_map[product_category],
        order_value,
        service_rating,
        refund_map[refund_requested]
    ])

    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("🚨 Delivery will likely be Delayed.")
    else:
        st.success("✅ Delivery is expected to be On Time.")
