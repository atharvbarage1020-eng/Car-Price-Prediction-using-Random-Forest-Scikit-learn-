import zipfile
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Path to your ZIP file ---
zip_path = "archive (2).zip"   # if in same folder, otherwise use full path

# --- Extract and load CSV from ZIP ---
with zipfile.ZipFile(zip_path, 'r') as z:
    # list files inside the zip
    print("Files inside ZIP:", z.namelist())
    # choose the CSV (edit if different name)
    with z.open("Car details v3.csv") as f:
        df = pd.read_csv(f)

# --- Prepare data ---
y = df["selling_price"]
X = df.drop(columns=["selling_price"])

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Preprocess numeric & categorical features ---
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(exclude=["int64", "float64"]).columns

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

# --- Build model pipeline ---
model = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
])

# --- Train model ---
model.fit(X_train, y_train)

# --- Evaluate model ---
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# --- Save model ---
with open("car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as car_price_model.pkl")
