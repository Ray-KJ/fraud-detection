import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Fraud Detective", layout="wide")

st.title("Fraud Detective: Real-Time Detection")
st.write("This dashboard connects to your Dockerized XGBoost API to analyze transaction risk.")

# --- SIDEBAR: Input Fields ---
st.sidebar.header("Transaction Details")
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
time = st.sidebar.slider("Seconds since first transaction", 0, 172792, 400)

# To make it easy, we'll let the user input the "V" features as a comma-separated list 
# or just generate random ones for testing.
v_features = st.sidebar.text_input("V1-V28 Features (comma separated)", "0.0,"*27 + "0.0")

if st.sidebar.button("Analyze Transaction"):
    # Prepare the payload for the API
    try:
        v_list = [float(x) for x in v_features.split(",")]
        payload = {"features": [time] + v_list + [amount]}
        
        #  CALLING YOUR DOCKER API
        response = requests.post("http://localhost:8000/predict", json=payload)
        prediction = response.json()["prediction"]
        probability = response.json()["fraud_probability"]

        # --- VISUALIZATION: Display Results ---
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error(" HIGH RISK: Potential Fraud Detected!")
            else:
                st.success(" LOW RISK: Transaction Appears Safe.")
            
            st.metric("Fraud Probability", f"{probability:.2%}")

        with col2:
            # Simple bar chart for the probability
            chart_data = pd.DataFrame({
                "Category": ["Safe", "Fraud"],
                "Probability": [1-probability, probability]
            })
            fig = px.bar(chart_data, x="Category", y="Probability", color="Category",
                         color_discrete_map={"Safe": "green", "Fraud": "red"})
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error connecting to API: {e}. Make sure your Docker container is running!")