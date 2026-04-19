import streamlit as st
import pickle
import numpy as np

# ── Load model files ──────────────────────────────────────
model     = pickle.load(open('model.pkl', 'rb'))
le_reason = pickle.load(open('le_reason.pkl', 'rb'))

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="911 Calls Predictor",
    page_icon="🚨",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────
st.title("🚨 911 Call Reason Predictor")
st.markdown("Enter the details below to predict whether a 911 call is **EMS**, **Fire**, or **Traffic**.")
st.divider()

# ── Input Form ────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    lat  = st.number_input("📍 Latitude",  
                            min_value=39.0, max_value=42.0, 
                            value=40.29, step=0.001, format="%.3f")
    lng  = st.number_input("📍 Longitude", 
                            min_value=-76.0, max_value=-74.0, 
                            value=-75.58, step=0.001, format="%.3f")
    hour = st.slider("🕐 Hour of Day", 0, 23, 12)

with col2:
    month = st.selectbox("📅 Month", options=list({
        1:'January', 2:'February', 3:'March', 4:'April',
        5:'May', 6:'June', 7:'July', 8:'August',
        9:'September', 10:'October', 11:'November', 12:'December'
    }.items()), format_func=lambda x: x[1])

    day = st.selectbox("📆 Day of Week", options=list({
        0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday',
        4:'Friday', 5:'Saturday', 6:'Sunday'
    }.items()), format_func=lambda x: x[1])

    year = st.selectbox("📅 Year", [2015, 2016, 2017, 2018, 2019, 2020])

st.divider()

# ── Predict Button ────────────────────────────────────────
if st.button("🔍 Predict Reason", use_container_width=True):
    
    # Prepare input — no scaling needed for Random Forest
    input_data = np.array([[lat, lng, day[0], month[0], hour, year]])
    
    # Predict
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    reason = le_reason.inverse_transform([prediction])[0]
    
    # ── Display result ────────────────────────────────────
    st.subheader("🎯 Prediction Result")
    
    color_map = {'EMS': '🔵', 'Fire': '🔴', 'Traffic': '🟢'}
    st.markdown(f"## {color_map[reason]} This is an **{reason}** call")
    
    st.subheader("📊 Confidence Scores")
    for i, cls in enumerate(le_reason.classes_):
        st.progress(float(probability[i]), 
                    text=f"{cls}: {probability[i]*100:.1f}%")