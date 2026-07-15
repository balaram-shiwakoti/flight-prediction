import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Generate realistic mock flight records
np.random.seed(42)
n_samples = 1500

data = {
    "airline": np.random.choice(["Qantas", "Emirates", "AirAsia", "SingaporeAir"], n_samples),
    "source": np.random.choice(["Sydney", "Melbourne", "Brisbane"], n_samples),
    "destination": np.random.choice(["London", "Singapore", "Tokyo"], n_samples),
    "duration_hours": np.random.uniform(3, 24, n_samples),
    "days_left": np.random.randint(1, 50, n_samples)
}

df_flights = pd.DataFrame(data)

# 2. Base formula for realistic market price calculation
base_price = 300
duration_multiplier = df_flights["duration_hours"] * 45
early_booking_discount = (50 - df_flights["days_left"]) * 4
airline_premium = df_flights["airline"].map({"Emirates": 350, "SingaporeAir": 250, "Qantas": 150, "AirAsia": 0})

df_flights["price"] = base_price + duration_multiplier + early_booking_discount + airline_premium
df_flights["price"] += np.random.normal(0, 50, n_samples) # Add real-world variance

# 3. Apply One-Hot Encoding
df_encoded = pd.get_dummies(df_flights, columns=["airline", "source", "destination"], drop_first=True)

X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("🎯 Model Training Complete!")
print(f"📊 Mean Absolute Error (MAE): ${mae:.2f}")
print(f"📈 R² Score: {r2:.4f}")

# 6. Bundle model & metadata into Pickle
flight_model_artifact = {
    "model": model,
    "feature_columns": list(X.columns),
    "airlines": ["AirAsia", "Emirates", "Qantas", "SingaporeAir"],
    "sources": ["Brisbane", "Melbourne", "Sydney"],
    "destinations": ["London", "Singapore", "Tokyo"]
}

with open("flight_predictor_model.pkl", "wb") as f:
    pickle.dump(flight_model_artifact, f)

print("🎉 Model successfully saved to 'flight_predictor_model.pkl'!")
