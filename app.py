import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- PAGE CONFIGURATION & THEME ---
st.set_page_config(page_title="SkyPrice AI", page_icon="✈️", layout="centered")

# Visual custom accent styles
st.markdown("""
    <style>
    .main-header { color: #1E3A8A; font-weight: bold; text-align: center; }
    .price-box { 
        background-color: #EFF6FF; 
        border: 2px solid #3B82F6; 
        padding: 20px; 
        border-radius: 12px; 
        text-align: center; 
        margin-top: 20px;
    }
    .price-text { color: #1D4ED8; font-size: 32px; font-weight: bold; }
    </style>
""", allow_output_mutation=False, unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>✈️ SkyPrice: AI Flight Fare Predictor</h1>", unsafe_allow_html=True)
st.write("Using a linear regression model trained on flight metrics to forecast real-time airfares based on booking urgency, airline tier, and route details.")
st.markdown("---")

# --- LOAD PICKLED MODEL ARTIFACT ---
@st.cache_resource
def load_flight_engine():
    with open("flight_predictor_model.pkl", "rb") as f:
        artifact = pickle.load(f)
    return artifact

try:
    artifact = load_flight_engine()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    airlines = artifact["airlines"]
    sources = artifact["sources"]
    destinations = artifact["destinations"]
except FileNotFoundError:
    st.error("⚠️ 'flight_predictor_model.pkl' not found! Please run your training script first.")
    st.stop()

# --- DESIGN USER INPUT FORM ---
st.subheader("📋 Plan Your Itinerary")

col1, col2 = st.columns(2)

with col1:
    selected_airline = st.selectbox("Preferred Airline", airlines)
    selected_source = st.selectbox("Departure City", sources)

with col2:
    # Filter destinations to prevent flying to the same city
    available_destinations = [d for d in destinations if d != selected_source]
    selected_destination = st.selectbox("Arrival City", available_destinations)
    
    # Simple travel estimation helpers
    default_duration = 14 if "London" in selected_destination else 8
    duration = st.slider("Flight Duration (Hours)", min_value=3, max_value=24, value=default_duration)

# Days left slider
days_left = st.slider("Booking Window (Days left before departure)", min_value=1, max_value=50, value=25, 
                      help="Fares spike dramatically the closer you get to departure date!")

st.markdown("---")

# --- LIVE MACHINE LEARNING INFERENCE ---
if st.button("🔮 Calculate Fare Prediction", use_container_width=True):
    # 1. Create empty matching matrix row
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0.0 # Set defaults to zero
    
    # 2. Map continuous values
    input_df.at[0, "duration_hours"] = float(duration)
    input_df.at[0, "days_left"] = float(days_left)
    
    # 3. Map One-Hot encoded binary values
    if f"airline_{selected_airline}" in input_df.columns:
        input_df.at[0, f"airline_{selected_airline}"] = 1.0
    if f"source_{selected_source}" in input_df.columns:
        input_df.at[0, f"source_{selected_source}"] = 1.0
    if f"destination_{selected_destination}" in input_df.columns:
        input_df.at[0, f"destination_{selected_destination}"] = 1.0
        
    # 4. Predict via Linear Regression Model weights
    predicted_price = model.predict(input_df)[0]
    
    # Clip any theoretical negative outliers just in case
    predicted_price = max(100.0, predicted_price)

    # 5. Output UI box
    st.markdown(f"""
        <div class="price-box">
            <p style="margin:0; font-size: 16px; color: #4B5563;">ESTIMATED AIRFARE COST</p>
            <p class="price-text">${predicted_price:,.2f} AUD</p>
            <p style="margin:0; font-size: 13px; color: #6B7280;">Predicted using multi-variable regression weight matrices.</p>
        </div>
    """, unsafe_allow_html=True)
