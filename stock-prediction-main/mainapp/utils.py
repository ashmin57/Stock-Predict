import pandas as pd

def preprocess_csv(file):
    """
    Function to preprocess the uploaded CSV file.
    
    Args:
        file: The uploaded CSV file.
    
    Returns:
        df: A preprocessed DataFrame ready for training/prediction.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)

    # Check for missing values and handle them
    if df.isnull().values.any():
        df = df.fillna(method='ffill')  # Forward fill to handle missing values
        # Alternatively, you could drop rows with missing values:
        # df = df.dropna()

    # Ensure the 'Date' column is in datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date if necessary
    df = df.sort_values(by='Date')

    # Select relevant columns for the model
    # Assuming you need 'Open', 'High', 'Low', 'Close', 'Volume', 'Percent Change'
    columns_of_interest = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Percent Change']
    df = df[columns_of_interest]

    # Normalize or scale data if necessary (optional)
    # Example: Min-Max Scaling
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])

    # Return the preprocessed DataFrame
    return df
