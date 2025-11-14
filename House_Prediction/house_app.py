import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("🏡 House Price Prediction App")

st.write("Enter house details below and get the predicted price:")

# Input fields
sqft_living = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=2000)
bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
bathrooms = st.slider("Number of Bathrooms", 1, 10, 2)

# Prediction
if st.button("Predict Price"):
    features = np.array([[sqft_living, bedrooms, bathrooms]])
    prediction = model.predict(features)[0]
    st.success(f"🏠 Predicted House Price: ${prediction:,.2f}")
