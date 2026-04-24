import streamlit as st
import numpy as np
import pickle
import json
import pandas as pd

# Load model & tools
model = pickle.load(open("final_model_xgb.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open("category_info.json", "r") as f:
    category_info = json.load(f)

st.title("🧠 Stress Level Prediction App")

st.write("Masukkan data berikut untuk memprediksi tingkat stres:")

# ========================
# FORM INPUT
# ========================
with st.form("prediction_form"):
    
    st.subheader("📊 Input Data")

    age = st.number_input("Age", min_value=0, step=1)
    daily_screen_time_hours = st.number_input("Daily Screen Time (hours)", min_value=0.0, step=0.1)
    phone_usage_before_sleep_minutes = st.number_input("Phone Usage Before Sleep (minutes)", min_value=0.0, step=1.0)
    sleep_duration_hours = st.number_input("Sleep Duration (hours)", min_value=0.0, step=0.1)
    sleep_quality_score = st.number_input("Sleep Quality Score (0-10)", min_value=0.0, max_value=10.0, step=0.1)
    caffeine_intake_cups = st.number_input("Caffeine Intake (cups/day)", min_value=0.0, step=1.0)
    physical_activity_minutes = st.number_input("Physical Activity (minutes/day)", min_value=0.0, step=1.0)
    notifications_received_per_day = st.number_input("Notifications per Day", min_value=0.0, step=1.0)
    mental_fatigue_score = st.number_input("Mental Fatigue Score (0-10)", min_value=0.0, max_value=10.0, step=0.1)

    gender = st.selectbox("Gender", category_info["gender"])
    occupation = st.selectbox("Occupation", category_info["occupation"])

    submitted = st.form_submit_button("🔍 Predict")

# ========================
# PREDIKSI
# ========================
if submitted:

    input_dict = {
        'age': age,
        'daily_screen_time_hours': daily_screen_time_hours,
        'phone_usage_before_sleep_minutes': phone_usage_before_sleep_minutes,
        'sleep_duration_hours': sleep_duration_hours,
        'sleep_quality_score': sleep_quality_score,
        'caffeine_intake_cups': caffeine_intake_cups,
        'physical_activity_minutes': physical_activity_minutes,
        'notifications_received_per_day': notifications_received_per_day,
        'mental_fatigue_score': mental_fatigue_score,
        'gender': gender,
        'occupation': occupation
    }

    input_df = pd.DataFrame([input_dict])

    # ✅ STEP 1: One-hot encoding
    for col, categories in category_info.items():
        for cat in categories:
            col_name = f"{col}_{cat}"
            input_df[col_name] = (input_df[col] == cat).astype(int)
        input_df.drop(columns=[col], inplace=True)

    # ✅ STEP 2: Reindex agar urutan & jumlah kolom sama persis dengan training
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # ✅ STEP 3: Scale data dengan aman
    # Otomatis mendeteksi kolom apa saja yang digunakan scaler saat training
    if hasattr(scaler, 'feature_names_in_'):
        scaler_cols = list(scaler.feature_names_in_)
        input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])
    else:
        # Fallback manual jika scaler disave sebagai array numpy (tanpa nama kolom)
        num_cols = [
            'age', 'daily_screen_time_hours', 'phone_usage_before_sleep_minutes',
            'sleep_duration_hours', 'sleep_quality_score', 'caffeine_intake_cups',
            'physical_activity_minutes', 'notifications_received_per_day', 'mental_fatigue_score'
        ]
        if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ == len(num_cols):
            input_df[num_cols] = scaler.transform(input_df[num_cols])
        else:
            # Jika benar-benar di-fit dengan seluruh kolom
            input_df[:] = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_df)
    result = round(float(prediction[0]), 2)

    st.success(f"🧠 Predicted Stress Level: {result}")
