import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom styling for a polished look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained KNN model
@st.cache_resource
def load_model():
    with open("knn.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading 'knn.pkl': {e}")
    st.stop()

# App Header
st.title("🎓 Student Performance & Exam Score Predictor")
st.markdown("Provide student details below to generate performance predictions using your trained **K-Nearest Neighbors Regressor** model.")
st.markdown("---")

# Input Form Layout
with st.form("prediction_form"):
    st.subheader("📋 Student Attributes & Study Habits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        course = st.selectbox("Course / Program", ["Computer Science", "Engineering", "Business", "Arts", "Science", "Other"])
        study_hours = st.slider("Daily Study Hours", min_value=0.0, max_value=16.0, value=4.0, step=0.5)
        class_attendance = st.slider("Class Attendance (%)", min_value=0, max_value=100, value=85, step=1)
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])

    with col2:
        sleep_hours = st.slider("Sleep Hours per Night", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
        sleep_quality = st.selectbox("Sleep Quality", ["Good", "Average", "Poor"])
        study_method = st.selectbox("Study Method", ["Self-study", "Online Videos", "Group Study", "Coaching"])
        facility_rating = st.slider("Facility Rating (1-5)", min_value=1, max_value=5, value=4, step=1)
        exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Moderate", "Hard"])

    st.markdown("---")
    submitted = st.form_submit_button("Predict Performance")

if submitted:
    # Construct input dataframe matching model features[cite: 1]
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'course': [course],
        'study_hours': [study_hours],
        'class_attendance': [class_attendance],
        'internet_access': [internet_access],
        'sleep_hours': [sleep_hours],
        'sleep_quality': [sleep_quality],
        'study_method': [study_method],
        'facility_rating': [facility_rating],
        'exam_difficulty': [exam_difficulty]
    })
    
    try:
        # Note: If your model was trained on encoded numeric features (e.g., Label Encoding or One-Hot Encoding)
        # rather than raw strings, ensure you apply the same preprocessing transformers here before predicting.
        prediction = model.predict(input_data)
        
        st.success("### Prediction Successful!")
        st.metric(label="Predicted Performance / Score", value=f"{prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("💡 **Deployment Tip:** If scikit-learn throws a categorical feature type mismatch, make sure categorical columns are encoded identically to how they were transformed during your model training phase.")
