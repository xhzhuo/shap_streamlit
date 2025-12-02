import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
data_path = r"test and log file put here\Advertising_Data.csv"
df = pd.read_csv(data_path)
target_col = df.columns[-1]
y = df[target_col].values
X = df[df.columns[:-1]].values

print(f"Target Column: {target_col}")
print(f"Target Mean: {y.mean():.2f}")
print(f"Target Median: {np.median(y):.2f}")

# Train model (same params as test script)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Check predictions
print("\nChecking predictions:")
# Check index 10 of X_test (used in test script)
pred_idx_10 = model.predict([X_test[10]])[0]
print(f"X_test[10] Prediction: {pred_idx_10:.2f}")

# Check if any prediction matches 7051.33
all_preds = model.predict(X)
diffs = np.abs(all_preds - 7051.33)
min_diff_idx = np.argmin(diffs)
print(f"\nClosest prediction to 7051.33 is at index {min_diff_idx}: {all_preds[min_diff_idx]:.2f}")

# Check if 7051.33 is in the actual values
diffs_y = np.abs(y - 7051.33)
min_diff_y_idx = np.argmin(diffs_y)
print(f"Closest actual value to 7051.33 is at index {min_diff_y_idx}: {y[min_diff_y_idx]:.2f}")
