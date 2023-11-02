# Import the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense
from keras.models import Sequential

# Load the data into a Pandas DataFrame
df = pd.read_csv('AthensGR_2022-01-01_to_2022-12-31.csv')

# Extract the relevant features from the DataFrame
features = df[['temp', 'humidity', 'sealevelpressure']].values

# Preprocess the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)

# Split the data into a training set and a test set
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]


# Convert the data into a format suitable for an LSTM network
def create_dataset(data, time_steps):
    X, Y = [], []
    for i in range(len(data) - time_steps - 1):
        a = data[i:(i + time_steps), :]
        X.append(a)
        Y.append(data[i + time_steps, 0])
    return np.array(X), np.array(Y)


time_steps = 24
X_train, Y_train = create_dataset(train_data, time_steps)
X_test, Y_test = create_dataset(test_data, time_steps)

# Make sure the input and output arrays have the same number of samples
X_test = X_test[:len(Y_test)]
Y_test = Y_test[:len(X_test)]

# Reshape the data for the LSTM network
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
Y_train = Y_train.reshape(-1, 1)



# Define the LSTM network model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))

# Compile the model with a loss function and an optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model on the training set
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)



# Evaluate the model on the test set
score = model.evaluate(X_test, Y_test, verbose=1)
print(f'Test score: {score:.3f}')

# Make predictions on the test set
predictions = model.predict(X_test)

# Invert the predictions and the test set back to the original scale
predictions = scaler.inverse_transform(predictions.reshape(-1, 3))
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 3))


# Calculate the mean absolute error
mae = np.mean(np.abs(predictions - Y_test))
print(f'Mean absolute error: {mae:.3f}')
