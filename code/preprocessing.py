import argparse
import os
import requests
import tempfile

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# Since we get a headerless CSV file, we specify the column names here.
feature_columns_names = [
    "id",
    "gender",
    "weight",
    "height",
    "ap_hi",
    "ap_low",
    "cholesterol",
    "gluc",
    "smoke", 
    "alco",
    "active"
]
label_column = "rings"

categorical_features_names = [
    "alco","active","smoke"
]

feature_columns_dtype = {
    "id": np.float64,
    "gender": np.float64,
    "weight": np.float64,
    "height": np.float64,
    "ap_hi": np.float64,
    "ap_low": np.float64,
    "cholesterol": np.float64,
    "gluc": np.float64,
    "smoke": np.float64, 
    "alco": np.float64,
    "active": np.float64
}

# label_column_dtype = {"rings": np.float64}
# this might be the one to 


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    
    # Local path where the file should be saved
    # local_path = base_dir + "/input/cardio-train.csv"
    
    # Accessing the whole raw dataset for preprocessing
    local_path = base_dir + "/input/cardio-raw-kaggle.csv"
    
    # Download the file from S3 to the local directory and convert to dataframe
    # sagemaker.s3.S3Downloader.download(s3_uri, local_path)
    df = pd.read_csv(local_path, sep=";")
    print("dataframe size / shape", df.size, df.shape)
    
    print("Dataset loaded successfully. Now filling missing values with the median...")
    df = df.fillna(df.median())  # Fill missing values
    
    print("Converting 'gender' column to numeric (1=male, 2=female) as needed...")
    df['gender'] = df['gender'].map({1: 0, 2: 1})  # Convert gender to numeric

    X = df.drop(columns=['cardio'])  # Features
    y = df['cardio']  # Target variable
    
    # Split into training (60%) and temporary (40%)
    print("Performing initial train-temp split with 60% training and 40% temporary...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Split the temporary set into validation (50%) and test (50%) sets
    print("Splitting temporary set into validation and test sets (each 20%)...")
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print("Train, validation, and test splits completed.")
    print("Training set size:", X_train.shape)
    print("Validation set size:", X_val.shape)
    print("Test set size:", X_test.shape)
    
    # Writing locally to project
    pd.DataFrame(X_train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(X_test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    pd.DataFrame(X_val).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)

    pd.DataFrame(y_train).to_csv(f"{base_dir}/train-labels/train.csv", header=False, index=False)
    pd.DataFrame(y_test).to_csv(f"{base_dir}/test-labels/test.csv", header=False, index=False)
    pd.DataFrame(y_val).to_csv(f"{base_dir}/validation-labels/validation.csv", header=False, index=False)
