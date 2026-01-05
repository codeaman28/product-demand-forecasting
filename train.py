import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
import os

# 1. Load dataset
df = pd.read_csv("data/sales.csv")

# 2. Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# 3. Feature engineering (extract useful info from date)
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# 4. Select input features and target
X = df[["store", "item", "month", "day"]]
y = df["sales"]

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Make predictions
predictions = model.predict(X_test)

# 8. Evaluate model
mae = mean_absolute_error(y_test, predictions)
print("Model MAE:", mae)

# 9. Save trained model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/demand_model.pkl")

print("Model trained and saved successfully.")
