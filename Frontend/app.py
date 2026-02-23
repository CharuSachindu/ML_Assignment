import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load Model and Feature Order
# -----------------------------

MODEL_PATH = "bike_price_model.cbm"
COLUMN_PATH = "model_columns.pkl"

# Load CatBoost model
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

# Load training feature columns
model_columns = joblib.load(COLUMN_PATH)

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🚲 Bike Price Prediction System")

st.write("Enter bike details to predict price")

# -----------------------------
# Input Fields (8 Features Only)
# -----------------------------

brand = st.text_input("Brand")
model_name = st.text_input("Model")
engine_capacity = st.number_input("Engine Capacity")
mileage = st.number_input("Mileage")
bike_type = st.selectbox("Bike Type", ["Motorbikes", "E-bikes", "Scooters"])
condition = st.selectbox("Condition", ["Brand New", "Used"])
location = st.selectbox("Location", [
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya",
    "Galle", "Matara", "Hambantota", "Jaffna", "Kilinochchi", "Mannar",
    "Vavuniya", "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla",
    "Monaragala", "Ratnapura", "Kegalle"
])
bike_age = st.number_input("Bike Age", min_value=0)

# -----------------------------
# Prediction Section
# -----------------------------

if st.button("Predict Price"):

    try:
        # Collect user input (must match training feature order)
        user_data = [
            brand,
            model_name,
            engine_capacity,
            mileage,
            bike_type,
            condition,
            location,
            bike_age
        ]

        # Create dataframe
        input_data = pd.DataFrame([user_data], columns=model_columns)

        # Convert categorical features to string (VERY IMPORTANT)
        input_data = input_data.astype(str)

        # Prediction
        prediction = model.predict(input_data)

        st.success(f"💰 Predicted Bike Price: {prediction[0]:,.2f}")

        # SHAP Explanation
        st.subheader("📊 Feature Importance Explanation")
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            
            # Force plot (shows how features push prediction up/down)
            st.write("**How each feature affects the prediction:**")
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                input_data.iloc[0],
                matplotlib=True
            )
            st.pyplot(plt.gcf())
            plt.close()
            
            # Waterfall plot (alternative view)
            st.write("**Feature Contribution Breakdown:**")
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=input_data.iloc[0],
                    feature_names=list(input_data.columns)
                ),
                show=False
            )
            st.pyplot(plt.gcf())
            plt.close()
            
        except Exception as shap_error:
            st.info(f"SHAP visualization unavailable: {str(shap_error)}")

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")