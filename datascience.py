import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Simulate Large Raw Data
np.random.seed(42)
n_samples = 100000  # Large dataset with 100k rows

raw_data = {
    "Year": np.random.randint(2000, 2024, size=n_samples),
    "GDP (Billion USD)": np.random.uniform(100, 500, size=n_samples),
    "Population (Million)": np.random.uniform(50, 300, size=n_samples),
    "Inflation (%)": np.random.uniform(1, 15, size=n_samples),
    "Unemployment (%)": np.random.uniform(3, 12, size=n_samples),
    "Exports (Billion USD)": np.random.uniform(5, 200, size=n_samples),
    "Imports (Billion USD)": np.random.uniform(10, 250, size=n_samples),
}

raw_df = pd.DataFrame(raw_data)

# Step 2: Data Cleaning and Feature Engineering
# Add new features like Trade Balance
raw_df["Trade Balance (Billion USD)"] = raw_df["Exports (Billion USD)"] - raw_df["Imports (Billion USD)"]

# Encode categorical variables if needed (e.g., Year into categories)
raw_df["Year_Category"] = pd.cut(raw_df["Year"], bins=[1999, 2009, 2019, 2024], labels=["2000-2009", "2010-2019", "2020-2024"])

# Step 3: Feature Selection
features = ["GDP (Billion USD)", "Population (Million)", "Inflation (%)", "Unemployment (%)", "Exports (Billion USD)", "Trade Balance (Billion USD)"]
target = "GDP (Billion USD)"

X = raw_df[features]
y = raw_df[target]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build an ML Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predictions and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")

# Step 7: Feature Importance
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Step 8: Visualization
plt.figure(figsize=(10, 6))
plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.title("Feature Importance in Predicting GDP")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Step 9: Save Processed Data for Further Analysis
raw_df.to_csv("processed_dataset.csv", index=False)
print("\nProcessed dataset saved as 'processed_dataset.csv'.")
