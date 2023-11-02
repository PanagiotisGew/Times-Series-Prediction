Weather Data Time Series Forecasting with LSTM

Author: Panagiotis Georgiadis

This Python code demonstrates how to use Long Short-Term Memory (LSTM) networks for time series forecasting with weather data. The example focuses on predicting weather parameters such as temperature, humidity, and sea-level pressure for a specific location.

Problem Description:
The problem addressed in this code is as follows:

We have historical weather data (temperature, humidity, and sea-level pressure) recorded at regular intervals.
The goal is to build an LSTM model to predict future weather conditions based on past data.

Dependencies:
This code relies on several Python libraries and modules, including:

numpy: Used for numerical operations.
pandas: Used for data handling and manipulation.
scikit-learn: Utilized for data preprocessing (MinMax scaling).
keras.layers.LSTM: Provides the LSTM layer for building the neural network.
keras.layers.Dense: Adds a fully connected dense layer to the model.
keras.models.Sequential: Allows the creation of a sequential model for deep learning.

How to Use:
To use this code for weather data forecasting, follow these steps:

Load Data: Prepare a CSV file containing historical weather data, including timestamps, temperature, humidity, and sea-level pressure. Specify the file path in the code.

Data Preprocessing: The code pre-processes the data by scaling it using MinMax scaling. Adjust the columns you want to predict by modifying the features variable.

LSTM Model Configuration: Configure the LSTM model by specifying the number of time steps, LSTM units, and other hyperparameters as needed.

Model Training: Train the LSTM model on the training set using the specified loss function and optimizer.

Model Evaluation: Evaluate the model's performance on the test set using metrics such as Mean Squared Error (MSE) to calculate the loss.

Generate Predictions: Generate weather parameter predictions on the test set using the trained model.

Post-processing: Invert the scaled predictions and the test set data to the original scale for better interpretability.

Performance Measurement: Calculate the Mean Absolute Error (MAE) to assess the model's accuracy.

Example Data:
This code includes an example CSV file containing weather data for Athens, Greece, from January 1, 2022, to December 31, 2022. You can replace it with your own data.

License:
This code is provided under a license that specifies the terms and conditions for its use and distribution.
Feel free to adapt and modify the code to work with your specific weather data and forecasting needs. Enjoy exploring time series forecasting with LSTM networks and making predictions based on historical weather data!
