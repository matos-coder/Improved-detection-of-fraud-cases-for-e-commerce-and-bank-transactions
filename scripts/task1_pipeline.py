# scripts/task1_pipeline.py

import pandas as pd
import ipaddress
from typing import Optional

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads data from a specified CSV file path.
    """
    try:
        print(f"🔄 Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully.")
        print(f"   Shape of the dataframe: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ ERROR: The file was not found at {file_path}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading the data: {e}")
        return None

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataframe by handling duplicates and converting data types.
    """
    print("\n--- Starting Data Cleaning and Preparation ---")
    
    print(f"Initial number of rows: {len(df)}")
    df.drop_duplicates(inplace=True)
    print(f"Number of rows after dropping duplicates: {len(df)}")
    
    print("🔄 Converting time columns to datetime objects...")
    try:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        print("✅ Time columns converted successfully.")
    except Exception as e:
        print(f"❌ An error occurred during time conversion: {e}")
        
    return df

def merge_with_ip_data(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges fraud data with geolocation data using the numeric IP address.
    """
    print("\n--- Merging with Geolocation Data ---")
    try:
        # --- THIS IS THE CORRECTED LOGIC ---
        # The 'ip_address' column is already a number (float), so we just convert it to an integer.
        print("🔄 Converting float 'ip_address' column to integer format...")
        fraud_df['ip_address_int'] = fraud_df['ip_address'].astype(int)
        print("✅ 'ip_address' converted to integer.")

        print("🔄 Merging dataframes based on IP range. This may take a moment...")
        country_list = []
        # This loop is for demonstration. For very large data, `pd.merge_asof` would be more efficient.
        for ip_int in fraud_df['ip_address_int']:
            country = ip_df.loc[
                (ip_df['lower_bound_ip_address'] <= ip_int) &
                (ip_df['upper_bound_ip_address'] >= ip_int),
                'country'
            ].values
            country_list.append(country[0] if len(country) > 0 else 'Unknown')
        
        fraud_df['country'] = country_list
        fraud_df.drop(columns=['ip_address_int'], inplace=True) # Clean up temporary column
        
        print(f"✅ Merge complete. Found {fraud_df['country'].nunique()} unique countries.")
        print(f"   Transactions with 'Unknown' country: {fraud_df[fraud_df['country'] == 'Unknown'].shape[0]}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during the merge: {e}")
        
    return fraud_df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A wrapper function to engineer all new features.
    """
    print("\n--- Starting Feature Engineering ---")
    try:
        # 1. Time since signup
        df['time_since_signup_seconds'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        print("✅ Feature 'time_since_signup_seconds' created.")
        
        # 2. Time-based features
        df['purchase_hour_of_day'] = df['purchase_time'].dt.hour
        df['purchase_day_of_week'] = df['purchase_time'].dt.dayofweek  # Monday=0, Sunday=6
        print("✅ Features 'purchase_hour_of_day' and 'purchase_day_of_week' created.")
        
        # 3. Frequency features
        df['device_id_count'] = df.groupby('device_id')['device_id'].transform('count')
        df['user_id_count'] = df.groupby('user_id')['user_id'].transform('count')
        print("✅ Features 'device_id_count' and 'user_id_count' created.")
    except Exception as e:
        print(f"❌ An error occurred during feature engineering: {e}")
        
    return df