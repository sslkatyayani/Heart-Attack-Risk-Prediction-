# Heart-Attack-Risk-Prediction-

This project implements a **Heart Disease Risk Prediction system** using machine learning.  
It classifies patients into **High Risk** or **Low Risk** categories based on health parameters.  
The model uses **K-Nearest Neighbors (KNN)** and **Naive Bayes (GaussianNB)** classifiers and compares their performance using standard evaluation metrics.

---

## Objectives

- Predict the risk level of heart disease using patient health data  
- Apply machine learning classification techniques  
- Compare model performance using **Accuracy**, **F1-Score** and **ROC-AUC**  
- Provide **user input–based risk prediction**

---

## Technologies Used

- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib  

---

## Dataset

The project uses the **Heart Disease dataset** from GitHub:

- URL: [Heart Dataset](https://github.com/lukaskris/data-science-heart-disease-prediction/blob/main/heart.csv)

**Target Column:** `HeartDisease`  
- `1` indicates presence of heart disease  
- `0` indicates no heart disease

**Features used:**

| Feature | Description |
|---------|-------------|
| Age | Patient age in years |
| RestingBP | Resting blood pressure (mmHg) |
| MaxHR | Maximum heart rate achieved during exercise |
| FastingBS | Fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes) |
| ExerciseAngina | Exercise-induced angina (0 = No, 1 = Yes) |

> Note: `ExerciseAngina` is converted from categorical `"Y"`/`"N"` to numeric `0`/`1`.

---

## How to Run the Project

1. Install required libraries:
pip install numpy pandas scikit-learn matplotlib
2.Run the program:
python prediction.py

## Output
