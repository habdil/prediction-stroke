import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Function untuk load model dan data
@st.cache_resource
def load_model():
    model = joblib.load('./data/models/optimized_stroke_model.joblib')
    metadata = joblib.load('./data/models/model_metadata.joblib')
    original_data = pd.read_csv('./data/processed/stroke_data_final.csv')
    return model, metadata, original_data

# Load model dan data
try:
    model, metadata, original_data = load_model()
    EXPECTED_COLUMNS = original_data.drop('stroke', axis=1).columns.tolist()
    optimal_threshold = metadata['optimized_performance']['optimal_threshold']
except Exception as e:
    st.error("Error loading model and data. Please check file paths.")
    st.stop()

# Functions untuk preprocessing
def create_features(data_dict):
    """Membuat fitur tambahan yang diperlukan"""
    # Konversi ke float untuk perhitungan
    for key in ['age', 'bmi', 'avg_glucose_level', 'hypertension', 'heart_disease']:
        data_dict[key] = float(data_dict[key])
    
    # Buat fitur tambahan
    data_dict['age_health_interaction'] = data_dict['age'] * (data_dict['hypertension'] + data_dict['heart_disease'])
    data_dict['bmi_glucose_risk'] = data_dict['bmi'] * data_dict['avg_glucose_level']
    
    # Hitung risk factors
    high_bmi = 1 if data_dict['bmi'] >= 25 else 0
    high_glucose = 1 if data_dict['avg_glucose_level'] >= 200 else 0
    data_dict['risk_factors'] = (data_dict['hypertension'] + 
                                data_dict['heart_disease'] + 
                                high_bmi + high_glucose)
    
    data_dict['age_lifestyle_risk'] = data_dict['age'] * data_dict['risk_factors']
    
    return data_dict

def prepare_input_data(data_dict):
    """Menyiapkan data input sesuai format yang diharapkan model"""
    # Buat fitur tambahan
    data_dict = create_features(data_dict)
    
    # Buat DataFrame dengan satu baris
    df = pd.DataFrame([data_dict])
    
    # Pastikan semua kolom yang diharapkan ada
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    
    # Urutkan kolom sesuai dengan urutan training
    df = df[EXPECTED_COLUMNS]
    
    return df

# Function untuk prediksi
def predict_stroke(model, data, threshold):
    """Melakukan prediksi stroke"""
    proba = model.predict_proba(data)[:, 1]
    predictions = (proba >= threshold).astype(int)
    return predictions, proba

# Main App
def main():
    # Title
    st.title("🏥 Stroke Prediction System")
    st.write("This system helps predict stroke risk based on patient information.")
    
    # Sidebar for patient information
    st.sidebar.header("Patient Information")
    
    # Input fields
    with st.sidebar.form("patient_form"):
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        gender = st.selectbox("Gender", ["Female", "Male"])
        
        col1, col2 = st.columns(2)
        with col1:
            hypertension = st.checkbox("Hypertension")
        with col2:
            heart_disease = st.checkbox("Heart Disease")
            
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=500.0, value=100.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
        
        smoking_status = st.selectbox("Smoking Status", 
            ["never smoked", "formerly smoked", "smokes", "Unknown"])
        
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        work_type = st.selectbox("Work Type", 
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        
        submitted = st.form_submit_button("Predict")
    
    # Jika form disubmit
    if submitted:
        # Prepare data dictionary
        data_dict = {
            'age': age,
            'gender': 1 if gender == "Male" else 0,
            'hypertension': 1 if hypertension else 0,
            'heart_disease': 1 if heart_disease else 0,
            'ever_married': 1 if ever_married == "Yes" else 0,
            'Residence_type': 1 if residence_type == "Urban" else 0,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'work_type_Govt_job': 1 if work_type == "Govt_job" else 0,
            'work_type_Never_worked': 1 if work_type == "Never_worked" else 0,
            'work_type_Private': 1 if work_type == "Private" else 0,
            'work_type_Self-employed': 1 if work_type == "Self-employed" else 0,
            'work_type_children': 1 if work_type == "children" else 0,
            'smoking_status_Unknown': 1 if smoking_status == "Unknown" else 0,
            'smoking_status_formerly smoked': 1 if smoking_status == "formerly smoked" else 0,
            'smoking_status_never smoked': 1 if smoking_status == "never smoked" else 0,
            'smoking_status_smokes': 1 if smoking_status == "smokes" else 0
        }
        
        # Process data and make prediction
        try:
            df = prepare_input_data(data_dict)
            pred, proba = predict_stroke(model, df, optimal_threshold)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Prediction Result")
                if pred[0] == 1:
                    st.error("⚠️ High Risk of Stroke")
                else:
                    st.success("✅ Low Risk of Stroke")
                
                st.metric("Stroke Probability", f"{proba[0]:.1%}")
            
            with col2:
                st.subheader("Risk Factors")
                risk_factors = []
                if hypertension:
                    risk_factors.append("• Hypertension")
                if heart_disease:
                    risk_factors.append("• Heart Disease")
                if bmi >= 25:
                    risk_factors.append("• High BMI")
                if avg_glucose_level >= 200:
                    risk_factors.append("• High Glucose Level")
                if age >= 65:
                    risk_factors.append("• Advanced Age")
                    
                if risk_factors:
                    for factor in risk_factors:
                        st.write(factor)
                else:
                    st.write("No major risk factors identified")
            
            with col3:
                st.subheader("Confidence Level")
                confidence_margin = abs(proba[0] - 0.5)
                if confidence_margin > 0.3:
                    st.write("High Confidence")
                elif confidence_margin > 0.15:
                    st.write("Medium Confidence")
                else:
                    st.write("Low Confidence")
            
            # Visualisasi
            st.subheader("Risk Visualization")
            fig, ax = plt.subplots(figsize=(10, 2))
            colors = ['green', 'yellow', 'red']
            plt.barh([0], [100], color='lightgray', alpha=0.3)
            plt.barh([0], [proba[0] * 100], color=colors[int(proba[0] * 2)])
            plt.axvline(x=optimal_threshold * 100, color='red', linestyle='--', alpha=0.5)
            plt.xlabel('Risk Percentage')
            plt.yticks([])
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    
    # Additional information
    with st.expander("See Explanation"):
        st.write("""
        This prediction system uses a machine learning model trained on stroke patient data. 
        The model considers various risk factors including:
        - Age
        - Medical conditions (hypertension, heart disease)
        - Lifestyle factors (smoking, BMI)
        - Blood glucose levels
        
        The prediction is based on statistical patterns and should not be used as the sole basis for medical decisions.
        Always consult with healthcare professionals for proper medical advice.
        """)

if __name__ == "__main__":
    main()