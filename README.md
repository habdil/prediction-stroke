# 🏥 Stroke Prediction System

## Overview
This project implements a machine learning system for predicting stroke risk based on patient health data. The system uses various health indicators and lifestyle factors to assess the probability of stroke occurrence.

## Project Structure
```
PREDICTION-STROKE/
├── data/
│   ├── models/
│   │   ├── best_stroke_model.joblib
│   │   ├── model_metadata.joblib
│   │   └── optimized_stroke_model.joblib
│   ├── processed/
│   │   ├── processed_stroke_data.csv
│   │   ├── stroke_data_engineered.csv
│   │   └── stroke_data_final.csv
│   └── raw/
│       └── healthcare-dataset-stroke-data.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── src/
│   ├── app.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── utils.py
└── README.md
```

## Key Features
- Data preprocessing and cleaning
- Advanced feature engineering
- Model training with multiple algorithms
- Hyperparameter optimization
- Interactive web interface for predictions
- Detailed risk factor analysis
- Confidence level assessment

## Technical Details

### Data Processing Pipeline
1. **Data Exploration** (`01_data_exploration.ipynb`)
   - Initial data analysis
   - Feature distribution analysis
   - Missing value detection

2. **Data Preprocessing** (`02_data_preprocessing.ipynb`)
   - Missing value handling
   - Outlier detection and treatment
   - Feature encoding
   - Data balancing

3. **Feature Engineering** (`03_feature_engineering.ipynb`)
   - Creation of new features
   - Feature selection
   - Feature importance analysis

4. **Model Training** (`04_model_training.ipynb`)
   - Multiple model training (Random Forest, XGBoost, Logistic Regression)
   - Cross-validation
   - Model optimization

5. **Model Evaluation** (`05_model_evaluation.ipynb`)
   - Performance metrics
   - Feature importance analysis
   - Model interpretation

### Model Performance
- Best Model: Random Forest
- ROC-AUC Score: 0.884
- Optimal Classification Threshold: 0.453

## Setup and Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/habdil/prediction-stroke.git
cd prediction-stroke
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Verify data structure:
- Ensure all data files are in their respective directories
- Check model files are present in the `data/models/` directory

## Usage

### Running the Web Application
```bash
streamlit run src/app.py
```

### Using the Notebooks
1. Start with data exploration:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

2. Follow the notebook sequence for the complete analysis pipeline

### Model Testing
Use `05_model_evaluation.ipynb` for:
- Testing individual cases
- Batch predictions
- Performance analysis

## Input Features
- Age
- Gender
- Hypertension status
- Heart disease status
- Average glucose level
- BMI
- Smoking status
- Work type
- Residence type
- Marital status

## Output Interpretation
The system provides:
- Stroke risk probability
- Binary prediction (High Risk/Low Risk)
- Confidence level
- Contributing risk factors
- Visual risk assessment

## Contributing
Feel free to contribute to this project by:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset source: [\[Healthcare Stroke Dataset\]](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Contributors and maintainers
- Open source community

---
*Note: This system is intended for research and educational purposes only. Always consult healthcare professionals for medical advice.*