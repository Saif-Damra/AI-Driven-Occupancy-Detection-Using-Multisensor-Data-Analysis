"""
Data Processing Module for Occupancy Detection
================================================
Handles data loading, cleaning, feature engineering,
and preprocessing for the occupancy estimation dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load the occupancy estimation dataset."""
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time features from Date and Time columns.
    
    After correlation analysis, only 'hours' is kept as it shows
    a meaningful relationship with occupancy patterns.
    """
    print("\n🕐 Engineering time features...")
    
    df["Date"] = pd.to_datetime(df["Date"])
    df["hours"] = df["Date"].dt.hour
    
    # Extract hour from Time column
    df[["hours", "_min", "_sec"]] = df["Time"].str.split(":", expand=True)
    df["hours"] = df["hours"].astype(int)
    
    # Drop Date, Time, and weak time features (year, month, day, minutes, seconds)
    # Based on correlation analysis, these features have no strong relationship
    df = df.drop(columns=["Date", "Time", "_min", "_sec"], errors="ignore")
    
    print("   Kept: hours (strong correlation with occupancy)")
    print("   Dropped: Date, Time, year, month, day, minutes, seconds (weak correlation)")
    
    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepare data for model training.
    
    Steps:
    1. Split features and target
    2. Scale features using StandardScaler
    3. Split into train/test sets
    
    Returns
    -------
    tuple: (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    print("\n🔧 Preparing data for training...")
    
    target_col = "Room_Occupancy_Count"
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    feature_names = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   Features: {len(feature_names)}")
    print(f"   Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"   Target distribution:\n{y.value_counts().sort_index().to_string()}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names


def process_data(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Complete data processing pipeline.
    
    Returns
    -------
    tuple: (X_train, X_test, y_train, y_test, scaler, feature_names, raw_df)
    """
    print("=" * 60)
    print("🚀 OCCUPANCY DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    # Load
    df = load_data(filepath)
    raw_df = df.copy()
    
    # Feature engineering
    df = engineer_time_features(df)
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"\n⚠️  Found {missing} missing values. Dropping rows...")
        df = df.dropna()
    else:
        print("\n✅ No missing values found")
    
    # Prepare for training
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
        df, test_size, random_state
    )
    
    print(f"\n✅ Processing complete!")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test, scaler, feature_names, raw_df
