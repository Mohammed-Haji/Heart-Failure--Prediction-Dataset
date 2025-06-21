# Heart Disease Prediction Project

## Project Overview
This project implements an end-to-end machine learning solution for predicting heart disease using a comprehensive dataset and a user-friendly Streamlit interface.

## Dataset Information
- **Source**: Kaggle - Heart Disease Prediction Dataset https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download
- **Size**: 270 observations with 13 features
- **Target Variable**: Heart disease presence (binary classification)
- **Features**: Age, sex, chest pain type, blood pressure, cholesterol, ECG results, and more

## Model Performance
- **Algorithm**: Logistic Regression
- **Training Accuracy**: 85.19%
- **Testing Accuracy**: 85.19%
- **Cross-validation Accuracy**: 82.88% (±13.64%)

## Project Structure
```
heart_disease_project/
├── data/
│   ├── dataset_heart.csv
│   └── processed/
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       └── scaler.pkl
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── gui_app.py
├── models/
│   ├── heart_disease_model.pkl
│   └── scaler.pkl
└── docs/
    └── screenshots/
```

## Files Description

### preprocess.py
- Loads and explores the dataset
- Handles data cleaning and preprocessing
- Splits data into training and testing sets
- Scales features using StandardScaler
- Saves preprocessed data for model training

### train_model.py
- Loads preprocessed data
- Trains a Logistic Regression model
- Evaluates model performance with various metrics
- Saves the trained model and scaler

### gui_app.py
- Streamlit web application for heart disease prediction
- Interactive interface for inputting patient data
- Real-time prediction with probability scores
- User-friendly visualization of results

## How to Run 

-- First Method 

1. **Data Preprocessing**:
   ```bash
   cd src
   python preprocess.py
   ```

2. **Model Training**:
   ```bash
   python train_model.py
   ```

3. **GUI Application**:
   ```bash
   streamlit run gui_app.py
   ```
-- Second Method (easiest way)

 **Open Google colabs**:
   1- create a new Notebook 
   
   2- load the (heart_disease_model.pkl)
  
   3- Go to Files click on the Upload to session storage then upload the Heart.csv file to load the whole dataset 

   4- Run all the codes and lastly go to the last line to display GUI, look for: 
  
  
   from pyngrok import ngrok

Make sure your REAL token is pasted here
NGROK_TOKEN = "2ySuexCwulNdlDwdLbkb7Bevdgl_6oBi6H6F7tjhPhYt2b8Fp"
ngrok.set_auth_token(NGROK_TOKEN)

Launch the app
public_url = ngrok.connect(8501)
print(f"Click this link to view your app: {public_url}")
!streamlit run gui_app.py
   ```

   Check Ngrok website to get NGROK_TOKEN most of the times you have same local TOKEN CODE which is "2ySuexCwulNdlDwdLbkb7Bevdgl_6oBi6H6F7tjhPhYt2b8Fp"

   5- After applying that it provides you a (Hosted website) click on the Visit you will be able to see the GUI 


## Features
- Clean, modular code structure
- Comprehensive data preprocessing
- Model evaluation with multiple metrics
- Interactive web interface
- Real-time predictions
- Risk assessment visualization

## Dependencies
- pandas
- numpy
- scikit-learn
- streamlit
- pickle

## Model Details
The Logistic Regression model uses 13 medical features to predict heart disease:
1. Age
2. Sex
3. Chest pain type
4. Resting blood pressure
5. Serum cholesterol
6. Fasting blood sugar
7. Resting ECG results
8. Maximum heart rate
9. Exercise induced angina
10. ST depression (oldpeak)
11. ST segment slope
12. Number of major vessels
13. Thalassemia

## Disclaimer
This application is for educational purposes only and should not replace professional medical advice. Always consult healthcare professionals for medical decisions.

