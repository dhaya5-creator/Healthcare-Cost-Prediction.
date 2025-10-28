# ==============================================================
# Project: Healthcare Cost Prediction using Regression
# SDG 3: Good Health & Well-Being
# Author: Your Name
# ==============================================================

# ✅ Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ✅ Load Dataset
# Make sure "insurance.csv" is in the same folder
data = pd.read_csv("insurance.csv")

# ✅ Encode Categorical Columns
encoder = LabelEncoder()
data['sex'] = encoder.fit_transform(data['sex'])
data['smoker'] = encoder.fit_transform(data['smoker'])
data['region'] = encoder.fit_transform(data['region'])

# ✅ Split Features & Target
X = data.drop('charges', axis=1)
y = data['charges']

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Prediction
y_pred = model.predict(X_test)

# ✅ Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# ✅ Show Prediction Comparison for First 10 Values
comparison = pd.DataFrame({
    'Actual Cost': y_test.values[:10],
    'Predicted Cost': y_pred[:10]
})
print("\n===== ACTUAL vs PREDICTED =====")
print(comparison)

# ✅ Visualization
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")
plt.title("Actual vs Predicted Healthcare Cost")
plt.grid(True)
plt.show()
