import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os

def fetch_data(csv_file_path= "C:/Users/elsingy/Documents/AMDARI DS/Internship/RFM/Data/Bank_Trust_Dataset.csv"):
    """Fetch data from CSV file"""
    
    # Check if file exists
    if not os.path.exists("C:/Users/elsingy/Documents/AMDARI DS/Internship/RFM/Data/Bank_Trust_Dataset.csv"):
        logging.error("CSV file not found: {}".format("C:/Users/elsingy/Documents/AMDARI DS/Internship/RFM/Data/Bank_Trust_Dataset.csv"))
        return None

    # Read CSV file
    try:
        data = pd.read_csv("C:/Users/elsingy/Documents/AMDARI DS/Internship/RFM/Data/Bank_Trust_Dataset.csv")
        print("Successfully loaded {} transactions from {}".format(len(data), "C:/Users/elsingy/Documents/AMDARI DS/Internship/RFM/Data/Bank_Trust_Dataset.csv"))
        return data

    except Exception as e:
        logging.error("Error loading data from CSV: {}".format(e))
        return None
    



def preprocess_data(data):
    """Preprocess the transaction data with consistency checks."""
    if data is None:
        print("No data available. Please fetch data first.")
        return None

    print("Starting Data Preprocessing...")
    
    df = data.copy()

    # Check for duplicates
    total_duplicates = df.duplicated().sum()
    transaction_duplicates = df['TransactionID'].duplicated().sum()
    print(f"The total duplicates is : {total_duplicates}")
    print(f"The total transaction duplicates is : {transaction_duplicates}")

    # Drop duplicated transaction IDs
    df = df.drop_duplicates(subset=['TransactionID'])

    # Count unique customers
    unique_customers = df['CustomerID'].nunique()
    print(f"Unique customers: {unique_customers}")

    # Drop missing values
    df = df.dropna()

    # Ensure proper datetime format
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')

    return df




def calculate_rfm_metrics(data):
    """Calculate RFM metrics and merge customer demographics."""
    if data is None:
        print("No data available. Please fetch and preprocess data first.")
        return None

    # Reference date is one day after the latest transaction
    reference_date = data['TransactionDate'].max() + pd.Timedelta(days=1)

    # RFM calculation
    rfm_df = data.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (reference_date - x.max()).days,  # Recency
        'TransactionID': 'count',                                      # Frequency
        'TransactionAmount': 'sum'                                     # Monetary
    }).reset_index()

    rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Customer demographics (first known values)
    customer_demographics = data.groupby('CustomerID').agg({
        'CustomerDOB': 'first',
        'CustGender': 'first',
        'CustLocation': 'first',
        'CustAccountBalance': 'last'
    }).reset_index()

    # Merge RFM with demographics
    rfm_df = rfm_df.merge(customer_demographics, on='CustomerID', how='left')

    return rfm_df