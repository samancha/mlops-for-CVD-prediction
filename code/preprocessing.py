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

    df = pd.read_csv(
        f"{base_dir}/train/cardio-train.csv",
        # header=True,
        # names=feature_columns_names + [label_column],
        # dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )

    numeric_features = list(feature_columns_names)
    
    # numeric_features = list(feature_columns_names)
    # numeric_features.remove("sex")

    # create the transforming steps
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ["sex"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print(df.head())

    # y = df.pop("rings")
    # X_pre = preprocess.fit_transform(df)
    # y_pre = y.to_numpy().reshape(len(y), 1)

    # X = np.concatenate((y_pre, X_pre), axis=1)

    # np.random.shuffle(X)

    # train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    # pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    # pd.DataFrame(validation).to_csv(
    #     f"{base_dir}/validation/validation.csv", header=False, index=False
    # )
    # pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)


    # print("Dataset loaded successfully. Now filling missing values with the median...")
    # df = df.fillna(df.median())  # Fill missing values

    # print("Converting 'gender' column to numeric (1=male, 2=female) as needed...")
    # df['gender'] = df['gender'].map({1: 0, 2: 1})  # Convert gender to numeric

    # # Splitting data
    # print("Splitting the data into features (X) and target (y)...")
    # X = df.drop(columns=['cardio'])  # Features
    # y = df['cardio']  # Target variable

    # print("Performing train-test split with 80% training and 20% testing...")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # print("Train-test split completed. Training set size:", X_train.shape, ", Testing set size:", X_test.shape)
    
    # TODO: we still need a validation dataset and then the to_csv function will put in sagemaker file system

    # pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    # pd.DataFrame(validation).to_csv(
    #     f"{base_dir}/validation/validation.csv", header=False, index=False
    # )
    # pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    
