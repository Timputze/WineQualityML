# Wine Quality Prediction Model

## Overview
This Python script is designed to predict the quality of wine based on several physicochemical properties. It utilizes a dataset containing various features of wines and their respective quality ratings. The script showcases data loading, exploratory data analysis, preprocessing, model training, evaluation, and prediction phases with a focus on applying machine learning techniques to solve a classification problem.

## Features
- **Data Loading**: Reads wine quality data from a CSV file.
- **Exploratory Data Analysis (EDA)**: Includes statistical summaries, information about data types, missing values analysis, and correlation matrix visualization.
- **Data Preprocessing**: Covers feature scaling and handling imbalanced datasets using SMOTE.
- **Model Training and Evaluation**: Employs multiple classifiers including Logistic Regression, MLPClassifier, RandomForestClassifier, XGBoost, and SVC. It evaluates models based on accuracy and selects the best-performing classifier.
- **Prediction**: Implements a function to predict wine quality based on user input utilizing the best classifier.

## Prerequisites
To run this script, you need Python installed on your machine along with the following libraries:
- Numpy
- Pandas
- Seaborn
- Matplotlib
- Imbalanced-Learn
- Scikit-Learn
- XGBoost

You can install these packages using pip:
```bash
pip install numpy pandas seaborn matplotlib imbalanced-learn scikit-learn xgboost
```

## Usage
1. Ensure you have the required dataset named `WineQuality.csv` in the correct path as specified in the script.
2. Run the script in a Python environment:
```bash
python test.py
```
3. Follow the prompts to input values for the specified wine features.
4. The script will output the predicted quality of the wine based on the input features.
