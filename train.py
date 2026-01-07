import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

# 1. Load the dataset
try:
    data = pd.read_csv('data/Medicalpremium.csv')
except FileNotFoundError:
    print("Error: 'data/Medicalpremium.csv' not found.")
    exit()

# 2. Clean the data
data = data.dropna()

# If your CSV uses 1/0 already, get_dummies won't change them.
# If your CSV uses "Yes"/"No", this will convert them to numbers.
data = pd.get_dummies(data, drop_first=True)

# 3. Define features (X) and target (y)
if 'PremiumPrice' not in data.columns:
    print("Error: Target column 'PremiumPrice' not found.")
    exit()

X = data.drop('PremiumPrice', axis=1)
y = data['PremiumPrice']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Initialize and Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predict and Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("--- Model Evaluation Results ---")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 7. Save the model
if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(model, 'model/model.pkl')
print("\nModel saved successfully in model/model.pkl")