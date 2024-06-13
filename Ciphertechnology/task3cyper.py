import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt

# Title of the app
st.title("Time Series Forecasting App")

# File upload
uploaded_file = st.file_uploader(r"D:\programming\programming\python\age&genderdetection\Pedestrian_Detection_OpenCV-master\Pedestrian_Detection_OpenCV-master\Alcohol_Sales.csv", type="csv")

if uploaded_file is not None:
    # Read the file into a dataframe
    df = pd.read_csv(uploaded_file, parse_dates=['DATE'], index_col='DATE')
    st.write("Data preview:")
    st.write(df.head())

    # Plot the data
    st.write("Alcohol Sales Over Time")
    st.line_chart(df['Sales'])

    # User inputs for the ARIMA model parameters
    p = st.number_input("Enter the value of p:", min_value=0, value=5)
    d = st.number_input("Enter the value of d:", min_value=0, value=1)
    q = st.number_input("Enter the value of q:", min_value=0, value=0)

    # Split the data into training and test sets
    train, test = df['Sales'][:-12], df['Sales'][-12:]

    # Fit the ARIMA model
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()

    # Forecast the next 12 months
    forecast = model_fit.forecast(steps=12)
    forecast.index = test.index

    # Plot the ARIMA forecast
    st.write("ARIMA Forecast")
    plt.figure(figsize=(14, 7))
    plt.plot(train, label='Training')
    plt.plot(test, label='Actual')
    plt.plot(forecast, label='Forecast')
    plt.legend()
    st.pyplot()

    # Calculate and display RMSE for ARIMA
    rmse_arima = sqrt(mean_squared_error(test, forecast))
    st.write(f'ARIMA RMSE: {rmse_arima}')

    # Prophet model
    df_prophet = df.reset_index().rename(columns={'DATE': 'ds', 'Sales': 'y'})

    # Initialize and fit the model
    model_prophet = Prophet()
    model_prophet.fit(df_prophet)

    # Forecast the next 12 months
    future = model_prophet.make_future_dataframe(periods=12, freq='M')
    forecast_prophet = model_prophet.predict(future)

    # Plot the Prophet forecast
    st.write("Prophet Forecast")
    fig = model_prophet.plot(forecast_prophet)
    st.pyplot(fig)

    # Calculate and display RMSE for Prophet
    forecast_prophet = forecast_prophet.set_index('ds')
    rmse_prophet = sqrt(mean_squared_error(df['Sales'][-12:], forecast_prophet['yhat'][-12:]))
    st.write(f'Prophet RMSE: {rmse_prophet}')

    # Plot components of the Prophet forecast
    st.write("Prophet Forecast Components")
    fig_components = model_prophet.plot_components(forecast_prophet)
    st.pyplot(fig_components)
