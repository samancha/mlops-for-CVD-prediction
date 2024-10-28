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
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_low",
    "cholesterol",
    "gluc",
    "smoke", 
    "alco",
    "active"
]
label_column = "cardio"

categorical_features_names = [
    "gender"
]

feature_columns_dtype = {
    "id": np.float64,
    "age": np.float64,
    "gender": np.float64,
    "height": np.float64,
    "weight": np.float64,
    "ap_hi": np.float64,
    "ap_low": np.float64,
    "cholesterol": np.float64,
    "gluc": np.float64,
    "smoke": np.float64, 
    "alco": np.float64,
    "active": np.float64
}

label_column_dtype = {"cardio": np.float64}

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    
    # Accessing the raw dataset with out headers for preprocessing
    local_path = base_dir + "/input/cardio-raw-no-header.csv"
    
    # Download the file from S3 to the local directory and convert to dataframe
    # sagemaker.s3.S3Downloader.download(s3_uri, local_path)
    df = pd.read_csv(local_path,
                     header=None,
                     names=feature_columns_names + [label_column],
                     dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype))

    numeric_features = list(feature_columns_names)
    numeric_features.remove("gender")
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ["gender"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    y = df.pop("cardio")
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)
    print("Shape of X_pre after preprocessing:", X_pre.shape)

    combined = np.hstack((y_pre, X_pre))  # For shuffling together
    np.random.shuffle(combined)
    
    # Separate again after shuffling
    y_pre = combined[:, 0]  # Target column
    X_pre = combined[:, 1:]  # Feature columns

    # train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])
    X_train, X_val, X_test = np.split(X_pre, [int(0.7 * len(X_pre)), int(0.85 * len(X_pre))])
    y_train, y_val, y_test = np.split(y_pre, [int(0.7 * len(y_pre)), int(0.85 * len(y_pre))])
    
    print("Train, validation, and test splits completed.")
    print("Training set size:", X_train.shape, " labels: ", y_train.shape)
    print("Validation set size:", X_val.shape, " labels:", y_val.shape)
    print("Test set size:", X_test.shape, " labels:", y_test.shape)

    train_combined = np.concatenate([y_train.reshape(-1, 1), X_train], axis=1)
    validation_combined = np.concatenate([y_val.reshape(-1, 1), X_val], axis=1)
    test_combined = np.concatenate([y_test.reshape(-1, 1), X_test], axis=1)

    
    # Writing locally to project
    pd.DataFrame(train_combined).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(test_combined).to_csv(f"{base_dir}/test/test.csv", header=True, index=False)
    pd.DataFrame(validation_combined).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)


    pd.DataFrame(x_test).to_csv(f"{base_dir}/test-x/test-x.csv", header=False, index=False)
    pd.DataFrame(y_test).to_csv(f"{base_dir}/test-label/test-label.csv", header=True, index=False)
    # pd.DataFrame(y_val).to_csv(f"{base_dir}/validation-label/validation-label.csv", header=False, index=False)
    # pd.DataFrame(y_train).to_csv(f"{base_dir}/train-label/train-label.csv", header=False, index=False)
    # pd.DataFrame(y_test).to_csv(f"{base_dir}/test-label/test-label.csv", header=True, index=False)
    # pd.DataFrame(y_val).to_csv(f"{base_dir}/validation-label/validation-label.csv", header=False, index=False)


    # y = df.pop("cardio")
    # X_pre = preprocess.fit_transform(df)
    # y_pre = y.to_numpy().reshape(len(y), 1)

    # X = np.concatenate((y_pre, X_pre), axis=1)

    # np.random.shuffle(X)
    # train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])
    
