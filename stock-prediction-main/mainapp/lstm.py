import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import math

def lstm_model(company):
    # Read the CSV file
    df = pd.read_csv(company)

    # Convert the Date column to a datetime object
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the dataframe by date
    df = df.sort_values('Date')

    # Convert '--' to 0 in the 'Percent Change' column
    df['Percent Change'] = df['Percent Change'].replace('--', 0)

    # Convert relevant columns to float
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Percent Change'] = df['Percent Change'].astype(float)

    # Extract the 'Close' column for prediction
    data = df['Close'].values.reshape(-1, 1)

    # Scale the data using Min-Max Scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Define the training and testing data sizes
    train_size = int(len(scaled_data) * 0.8)
    test_size = len(scaled_data) - train_size

    # Split the data into training and testing sets
    train_data = scaled_data[:train_size, :]
    test_data = scaled_data[train_size:, :]

    def prepare_data(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps - 7):
            X.append(data[i:i + time_steps, 0])
            y.append(data[i + time_steps:i + time_steps + 7, 0])
        return np.array(X), np.array(y)

    # Define the number of time steps
    time_steps = 7

    # Prepare the training data
    X_train, y_train = prepare_data(train_data, time_steps)

    # Prepare the testing data
    X_test, y_test = prepare_data(test_data, time_steps)

    # Reshape the data for LSTM (samples, time_steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define LSTM parameters
    input_dim = 1  # Number of features (Close price)
    hidden_units = 100  # Number of LSTM units
    output_dim = 7  # Number of outputs (next 7 days)

    # Initialize weights with the correct shape
    Wf = np.random.rand(hidden_units, hidden_units + input_dim)  # Forget gate
    Wi = np.random.rand(hidden_units, hidden_units + input_dim)  # Input gate
    Wc = np.random.rand(hidden_units, hidden_units + input_dim)  # Cell state
    Wo = np.random.rand(hidden_units, hidden_units + input_dim)  # Output gate

    bf = np.random.rand(hidden_units)  # Forget gate bias
    bi = np.random.rand(hidden_units)  # Input gate bias
    bc = np.random.rand(hidden_units)  # Cell state bias
    bo = np.random.rand(hidden_units)  # Output gate bias

    # LSTM cell states and hidden states
    h_t = np.zeros(hidden_units)  # Changed to 1D array
    c_t = np.zeros(hidden_units)  # Changed to 1D array

    # Prepare the input for the LSTM
    predictions = []

    for i in range(X_train.shape[0]):
        x_t = X_train[i]  # Current input
        x_t = x_t.reshape(-1)  # Reshape input to (time_steps,)

        # Combine input and hidden state
        combined = np.hstack((h_t, x_t))  # (hidden_units + input_dim,)

        # Debug: Print shapes
        print(f"Iteration: {i}, combined shape: {combined.shape}")

        # Forget gate
        f_t = 1 / (1 + np.exp(-(combined @ Wf.T + bf)))  # Wf.T for correct shape
        print(f"f_t shape: {f_t.shape}")

        # Input gate
        i_t = 1 / (1 + np.exp(-(combined @ Wi.T + bi)))  # Wi.T for correct shape
        print(f"i_t shape: {i_t.shape}")

        # Cell state
        C_hat_t = np.tanh(combined @ Wc.T + bc)  # Wc.T for correct shape
        print(f"C_hat_t shape: {C_hat_t.shape}")

        # Update cell state
        c_t = f_t * c_t + i_t * C_hat_t

        # Output gate
        o_t = 1 / (1 + np.exp(-(combined @ Wo.T + bo)))  # Wo.T for correct shape
        print(f"o_t shape: {o_t.shape}")

        # Update hidden state
        h_t = o_t * np.tanh(c_t)

        # Store predictions for the last timestep
        if i == X_train.shape[0] - 1:
            predictions.append(h_t)

    # Reshape the output for the final prediction
    predicted_close_prices = np.array(predictions).flatten()

    # Inverse scale the predicted prices
    predicted_close_prices = scaler.inverse_transform(predicted_close_prices.reshape(-1, 1))

    # Prepare dates for predictions
    last_date = df['Date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=7, freq='D')
    df_predictions = pd.DataFrame({'close_price': predicted_close_prices.flatten(), 'date': forecast_dates})

    # Calculate actual close prices for the last 7 days
    actual_close_prices = scaler.inverse_transform(test_data[-7:, :])

    # Calculate MSE
    mse = mean_squared_error(actual_close_prices, predicted_close_prices)
    print("Mean Squared Error (MSE):", mse)

    # Calculate RMSE
    rmse = math.sqrt(mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Calculate R2 score
    r2 = r2_score(actual_close_prices, predicted_close_prices)
    print("R-squared (R2) score:", r2)

    print(df_predictions)
    return df_predictions
