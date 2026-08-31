import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("data/dataset.csv")


# Feature engineering
df['date'] = pd.to_datetime(df['date'])

df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday

df = df.drop('date', axis=1)


# Separate features and target
X = df.drop('Appliances', axis=1)
y = df['Appliances']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 1. Linear Regression
# --------------------------------------------------

lr_model = LinearRegression()

lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Results")
print("MAE:", mae_lr)
print("RMSE:", rmse_lr)
print("R²:", r2_lr)


# --------------------------------------------------
# 2. Random Forest Regressor
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Results")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)
print("R²:", r2_rf)


# --------------------------------------------------
# 3. Gradient Boosting Regressor
# --------------------------------------------------

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)

print("\nGradient Boosting Results")
print("MAE:", mae_gb)
print("RMSE:", rmse_gb)
print("R²:", r2_gb)


# --------------------------------------------------
# 4. Artificial Neural Network
# --------------------------------------------------

ann_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42
)

ann_model.fit(X_train_scaled, y_train)

y_pred_ann = ann_model.predict(X_test_scaled)

mae_ann = mean_absolute_error(y_test, y_pred_ann)
rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))
r2_ann = r2_score(y_test, y_pred_ann)

print("\nANN Results")
print("MAE:", mae_ann)
print("RMSE:", rmse_ann)
print("R²:", r2_ann)


# --------------------------------------------------
# Model Comparison
# --------------------------------------------------

results = {
    'Model': [
        'Linear Regression',
        'Random Forest',
        'Gradient Boosting',
        'ANN'
    ],
    'MAE': [
        mae_lr,
        mae_rf,
        mae_gb,
        mae_ann
    ],
    'RMSE': [
        rmse_lr,
        rmse_rf,
        rmse_gb,
        rmse_ann
    ],
    'R2': [
        r2_lr,
        r2_rf,
        r2_gb,
        r2_ann
    ]
}

results_df = pd.DataFrame(results)

print("\nModel Comparison")
print(results_df)
