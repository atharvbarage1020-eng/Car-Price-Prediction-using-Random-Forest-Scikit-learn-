# 🚗 Car Price Prediction using Random Forest

This project predicts the selling price of used cars using Machine Learning. It utilizes the **Car Details v3** dataset, performs data preprocessing, trains a **Random Forest Regressor**, evaluates model performance, and saves the trained model for future predictions.

## 📌 Features

- Load dataset directly from a ZIP file
- Handle missing values using SimpleImputer
- Encode categorical features using OneHotEncoder
- Train a Random Forest Regression model
- Evaluate performance using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- Save the trained model as a `.pkl` file

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Pickle
- ZipFile

## 📂 Project Structure

```
Car-Price-Prediction/
│── archive (2).zip
│── car_price_prediction.py
│── car_price_model.pkl
│── README.md
```

## 📊 Machine Learning Workflow

1. Load dataset from ZIP archive
2. Separate features and target variable
3. Split data into training and testing sets
4. Preprocess numerical and categorical data
5. Train Random Forest Regressor
6. Evaluate model performance
7. Save the trained model

## 📈 Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
```

Install dependencies:

```bash
pip install pandas scikit-learn
```

Run the project:

```bash
python car_price_prediction.py
```

## 💾 Output

The trained model is saved as:

```
car_price_model.pkl
```

This model can be loaded later for predicting car prices without retraining.

## 📚 Dataset

Dataset: **Car Details v3**

The dataset contains information such as:
- Car Name
- Year
- Selling Price
- Fuel Type
- Seller Type
- Transmission
- Owner
- Mileage
- Engine
- Max Power
- Seats

## 🚀 Future Improvements

- Hyperparameter tuning
- Feature engineering
- Flask/Streamlit web application
- XGBoost and LightGBM comparison
- Model deployment

## 👨‍💻 Author

**Atharv Barage**

B.Tech Artificial Intelligence & Machine Learning
