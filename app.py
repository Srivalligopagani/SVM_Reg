import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="SVR House Price", layout="centered")

st.title("House Price Prediction using SVR ")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Real estate.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------------------------------
# FEATURES
# ---------------------------------------------------
X = df.drop("Y house price of unit area", axis=1)
y = df["Y house price of unit area"]

# ---------------------------------------------------
# TRAIN MODEL (every run)
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

model = SVR(kernel="rbf")
model.fit(X_train, y_train)

# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("R2", f"{r2:.3f}")

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
st.subheader("Predict House Price")

feature_cols = X.columns

user_input = []

for col in feature_cols:
    val = st.number_input(col, value=float(df[col].mean()))
    user_input.append(val)

input_df = pd.DataFrame([user_input], columns=feature_cols)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
input_scaled = scaler_X.transform(input_df)

pred_scaled = model.predict(input_scaled)

prediction = scaler_y.inverse_transform(
    pred_scaled.reshape(-1, 1)
)[0][0]

st.success(f"Predicted House Price: {prediction:.2f}")