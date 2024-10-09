import pandas as pd

def preprocess_csv(file):
    df = pd.read_csv(file)

    # Remove commas from numeric columns such as 'Volume'
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].astype(str).str.replace(',', '')

    # Convert 'Volume' back to numeric (after removing commas)
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # Drop non-numeric columns like 'Date' and 'Symbol'
    df_numeric = df.drop(columns=['Date', 'Symbol'], errors='ignore')

    # Standardize only the numeric columns
    df_numeric = (df_numeric - df_numeric.mean()) / df_numeric.std()

    # Add back the 'Symbol' and 'Date' columns
    df['Symbol'] = df['Symbol']
    df['Date'] = df['Date']
    
    # Update the DataFrame with the standardized numeric columns
    df.update(df_numeric)

    return df
