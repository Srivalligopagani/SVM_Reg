import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction using SVR",
    layout="centered"
)

st.title("House Price Prediction using Support Vector Regressor")

st.write(
    "Predict House Price using SVR"
)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("Real estate.csv")

df = load_data()

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.subheader("Dataset Preview")

st.dataframe(df.head())

# ---------------------------------------------------
# FEATURES & TARGET
# ---------------------------------------------------

X = df.drop("Y house price of unit area", axis=1)

y = df["Y house price of unit area"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(
    y_train.values.reshape(-1, 1)
).ravel()

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

model = SVR(kernel='rbf')

model.fit(X_train, y_train)

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------

y_pred_scaled = model.predict(X_test)

y_pred = scaler_y.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
)

# ---------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

# ---------------------------------------------------
# GRAPH
# ---------------------------------------------------

st.subheader("Actual vs Predicted Prices")

fig, ax = plt.subplots()

ax.scatter(y_test, y_pred)

ax.set_xlabel("Actual Prices")

ax.set_ylabel("Predicted Prices")

ax.set_title("Actual vs Predicted")

st.pyplot(fig)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

st.subheader("Model Performance")

c1, c2 = st.columns(2)

c1.metric(
    "Mean Absolute Error",
    f"{mae:.2f}"
)

c2.metric(
    "Root Mean Squared Error",
    f"{rmse:.2f}"
)

c3, c4 = st.columns(2)

c3.metric(
    "R2 Score",
    f"{r2:.3f}"
)

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------

st.subheader("Predict House Price")

transaction_date = st.number_input(
    "Transaction Date",
    value=float(df["X1 transaction date"].mean())
)

house_age = st.number_input(
    "House Age",
    value=float(df["X2 house age"].mean())
)

mrt_distance = st.number_input(
    "Distance to Nearest MRT Station",
    value=float(df["X3 distance to the nearest MRT station"].mean())
)

stores = st.number_input(
    "Number of Convenience Stores",
    value=float(df["X4 number of convenience stores"].mean())
)

latitude = st.number_input(
    "Latitude",
    value=float(df["X5 latitude"].mean())
)

longitude = st.number_input(
    "Longitude",
    value=float(df["X6 longitude"].mean())
)

number = st.number_input(
    "Serial Number",
    value=float(df["No"].mean())
)

# ---------------------------------------------------
# INPUT ARRAY
# ---------------------------------------------------

input_data = np.array([[
    number,
    transaction_date,
    house_age,
    mrt_distance,
    stores,
    latitude,
    longitude
]])

# ---------------------------------------------------
# SCALE INPUT
# ---------------------------------------------------

input_scaled = scaler_X.transform(input_data)

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------

prediction_scaled = model.predict(input_scaled)

prediction = scaler_y.inverse_transform(
    prediction_scaled.reshape(-1, 1)
)[0][0]

# ---------------------------------------------------
# OUTPUT
# ---------------------------------------------------

st.success(
    f"Predicted House Price : {prediction:.2f}"
)