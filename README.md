# CodeAlpha_DieasePrediction

**Disease Prediction from Medical Data**

Project Overview
This project focuses on predicting the possibility of diseases using structured medical data and machine learning classification techniques. Patient attributes such as age, symptoms, and medical test results are analyzed to determine whether a disease is present.

The project is designed for academic mini-projects, internships, and practical machine learning applications.

Objective
To build and compare multiple machine learning classification models for disease prediction and identify the best-performing algorithm based on evaluation metrics.

Dataset Used
Heart Disease Dataset (UCI Machine Learning Repository)

- Total records: 1025
- Number of features: 13
- Target column: target
  - 0 → No disease
  - 1 → Disease present

Features Used
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression
- Slope of ST segment
- Number of major vessels
- Thalassemia

Algorithms Implemented
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
- XGBoost Classifier

Methodology
1. Load and explore the dataset
2. Preprocess data and apply feature scaling
3. Split the dataset into training and testing sets
4. Train multiple classification models
5. Evaluate models using performance metrics
6. Compare algorithms using graphical visualizations

Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

Results
Among all the implemented models, Random Forest and XGBoost achieved the highest accuracy. XGBoost was identified as the best-performing model for heart disease prediction due to its superior classification performance.

Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Google Colab

Future Scope
- Incorporate additional datasets such as Diabetes and Breast Cancer
- Deploy the model using web frameworks like Streamlit or Flask
- Improve performance using hyperparameter tuning
- Implement real-time prediction functionality

Disclaimer
This project is intended for educational purposes only and should not be used as a substitute for professional medical diagnosis.
