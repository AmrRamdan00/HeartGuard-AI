import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
import time

# --- Page Config ---
st.set_page_config(
    page_title="HeartGuard AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Modern" Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        height: 3em;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #d43f3f;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Model Training & Caching ---
# We use @st.cache_resource so the model only trains ONCE when the app starts,
# not every time you click a button.
@st.cache_resource
def load_and_train_model():
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.text("Loading dataset...")
    try:
        data = pd.read_csv('heart.csv')
    except FileNotFoundError:
        st.error("Error: 'heart.csv' not found. Please place it in the same directory.")
        st.stop()
        
    progress_bar.progress(20)
    
    # Preprocessing
    cols_to_convert = ['FastingBS', 'MaxHR', 'Oldpeak']
    for col in cols_to_convert:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    
    binary_cols = ['Sex', 'ExerciseAngina', 'FastingBS']
    le = LabelEncoder()
    for col in binary_cols:
        data[col] = le.fit_transform(data[col])
        
    multi_cols = ['ChestPainType', 'RestingECG', 'ST_Slope']
    data = pd.get_dummies(data, columns=multi_cols)
    
    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']
    
    # Save column structure for prediction alignment
    model_columns = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    progress_bar.progress(50)
    status_text.text("Training Neural Network (this happens once)...")
    
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Reduced verbosity for web app
    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
              epochs=100, batch_size=32, verbose=0, callbacks=[early_stop])
    
    progress_bar.progress(100)
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return model, scaler, model_columns

# Load the model
model, scaler, model_columns = load_and_train_model()

# --- 2. Sidebar UI ---
st.sidebar.title("🫀 Patient Data")
st.sidebar.markdown("Configure the patient's health parameters below.")

# Input Group 1: Demographics
st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age", 18, 100, 50)
sex = st.sidebar.radio("Sex", ["Male", "Female"], horizontal=True)

# Input Group 2: Vitals
st.sidebar.subheader("Vitals")
resting_bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.sidebar.slider("Cholesterol (mm/dl)", 100, 600, 200)
max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)

# Input Group 3: History & Conditions
st.sidebar.subheader("Conditions")
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
exercise_angina = st.sidebar.selectbox("Exercise Induced Angina?", ["No", "Yes"])
oldpeak = st.sidebar.number_input("Oldpeak (ST depression)", 0.0, 6.0, 0.0, step=0.1)

# Input Group 4: ECG Features
st.sidebar.subheader("ECG Parameters")
chest_pain = st.sidebar.selectbox(
    "Chest Pain Type", 
    ["ASY: Asymptomatic", "ATA: Atypical Angina", "NAP: Non-Anginal Pain", "TA: Typical Angina"]
)
resting_ecg = st.sidebar.selectbox(
    "Resting ECG Results", 
    ["Normal", "LVH: Left Ventricular Hypertrophy", "ST: ST-T Wave Abnormality"]
)
st_slope = st.sidebar.selectbox(
    "ST Slope", 
    ["Flat", "Up: Upsloping", "Down: Downsloping"]
)

# --- 3. Main Interface & Logic ---

st.title("🫀 HeartGuard AI Prediction System")
st.markdown("### Neural Network-Based Heart Disease Risk Assessment")
st.write("Adjust the values in the sidebar to predict the risk profile.")

st.divider()

# --- Prediction Logic ---
if st.sidebar.button("Analyze Risk Profile"):
    with st.spinner("Processing Neural Network..."):
        # 1. Manual Encoding Matching (Same logic as your script)
        sex_val = 1 if sex == "Male" else 0
        ex_angina_val = 1 if exercise_angina == "Yes" else 0
        fasting_bs_val = 1 if fasting_bs == "Yes" else 0
        
        # Parse complex strings (e.g., "ASY: Asymptomatic" -> "ASY")
        cp_code = chest_pain.split(":")[0]
        ecg_code = resting_ecg.split(":")[0]
        slope_code = st_slope.split(":")[0]
        
        # Create a dictionary for the input
        input_data = {
            'Age': [age],
            'Sex': [sex_val],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs_val],
            'MaxHR': [max_hr],
            'ExerciseAngina': [ex_angina_val],
            'Oldpeak': [oldpeak],
        }
        
        # Create DF
        input_df = pd.DataFrame(input_data)
        
        # Add the One-Hot Encoded columns manually initialized to 0
        # We look at the trained model_columns to know exactly what the model expects
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Set the specific One-Hot bits to 1 based on selection
        # Note: This relies on the column names generated by get_dummies in the training phase
        if f'ChestPainType_{cp_code}' in input_df.columns:
            input_df[f'ChestPainType_{cp_code}'] = 1
        if f'RestingECG_{ecg_code}' in input_df.columns:
            input_df[f'RestingECG_{ecg_code}'] = 1
        if f'ST_Slope_{slope_code}' in input_df.columns:
            input_df[f'ST_Slope_{slope_code}'] = 1
            
        # Ensure column order matches training EXACTLY
        input_df = input_df[model_columns]
        
        # Scale
        X_new_scaled = scaler.transform(input_df)
        
        # Predict
        prediction_prob = model.predict(X_new_scaled)[0][0]
        prob_percentage = prediction_prob * 100
        
        # --- Display Results ---
        st.subheader("Analysis Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Gauge Chart / Metric
            st.metric(label="Risk Probability", value=f"{prob_percentage:.1f}%")
        
        with col2:
            if prediction_prob >= 0.7:
                st.error("⚠️ HIGH RISK DETECTED")
                st.write("**Recommendation:** Consult a cardiologist immediately. Monitor vitals daily.")
            elif prediction_prob >= 0.5:
                st.warning("⚖️ MEDIUM RISK DETECTED")
                st.write("**Recommendation:** Lifestyle changes recommended (Diet/Exercise). Schedule a checkup.")
            else:
                st.success("✅ LOW RISK")
                st.write("**Recommendation:** Maintain healthy habits. Regular annual screening.")

        # Visual Bar
        st.write("Risk Scale:")
        st.progress(int(prob_percentage))
        st.caption("0% (Healthy) ----------------------------------> 100% (High Risk)")

else:
    st.info("👈 Please configure patient data in the sidebar and click **Analyze Risk Profile**.")

st.divider()
st.caption("Model: Keras Sequential Neural Network | Dataset: Heart Failure Prediction")