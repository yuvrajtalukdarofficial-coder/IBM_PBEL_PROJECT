# Smart Home Energy Consumption Prediction

## Project Overview

This project focuses on predicting household appliance energy consumption using Machine Learning.

The project follows a complete Machine Learning workflow, including data preprocessing, Exploratory Data Analysis (EDA), feature preparation, model training, evaluation, model comparison, error analysis, and final prediction.

## Problem Statement

Household appliance energy consumption depends on several environmental and sensor-related conditions.

The objective of this project is to develop a Machine Learning regression system that can predict appliance energy consumption from available environmental and sensor measurements.

## Objectives

- Analyze household energy-consumption data.
- Perform data preprocessing and cleaning.
- Explore relationships between features and appliance energy consumption.
- Train multiple regression models.
- Compare model performance using MAE, RMSE, and R².
- Analyze prediction errors and overfitting.
- Demonstrate prediction using unseen data.
- Identify the best-performing model.

## Dataset

The project uses the **Appliances Energy Prediction Dataset**.

The dataset contains household appliance energy-consumption measurements along with environmental and sensor-related features.

### Target Variable

`Appliances`

The target variable represents household appliance energy consumption.

### Important Features

The dataset contains measurements related to:

- Temperature
- Humidity
- Lighting
- Outdoor environmental conditions
- Weather-related variables
- Other sensor measurements

## Machine Learning Models

Four regression models were implemented and compared:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**
4. **Artificial Neural Network (ANN)**

The project guide requires these four models for regression projects and requires evaluation using MAE, RMSE, and R². :contentReference[oaicite:1]{index=1}

## Evaluation Metrics

The models are evaluated using:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

### RMSE — Root Mean Squared Error

Measures prediction error while giving greater importance to larger errors.

### R² — R-Squared

Measures how well the model explains the variation in the target variable.

## Exploratory Data Analysis

EDA was performed to understand the dataset and identify important patterns.

The visualizations include:

- Target-variable distribution
- Correlation analysis
- Feature relationship plots
- Energy-consumption patterns
- Actual vs Predicted values
- Model performance comparison
- Prediction-error analysis

The project guide requires a target distribution, correlation/relationship visualization, a domain-specific insight chart, model comparison, and an actual-vs-predicted plot for regression projects. :contentReference[oaicite:2]{index=2}

## Model Comparison

The models are compared using:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | — | — | — |
| Random Forest Regressor | — | — | — |
| Gradient Boosting Regressor | — | — | — |
| Artificial Neural Network | — | — | — |

The final values in this table should match the actual results obtained by running the project notebook.

## Best Model

The best-performing model is selected based on:

- Lower MAE
- Lower RMSE
- Higher R²

The model comparison is used to determine which approach provides the strongest performance on the held-out test data.

## Overfitting Analysis

Training and testing performance are compared to identify possible overfitting.

If the model performs substantially better on the training data than on unseen testing data, this indicates that the model may have learned patterns specific to the training data and may not generalize as well to new observations.

## Final Prediction

The trained model is used to predict appliance energy consumption for an unseen test sample.

The predicted value is compared with the actual value to demonstrate how the regression system performs on new data.

## Technologies Used

- Python
- Jupyter Notebook / Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras

## Project Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Preparation
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Error & Overfitting Analysis
   ↓
Final Prediction
