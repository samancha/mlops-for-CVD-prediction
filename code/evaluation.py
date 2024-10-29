import json
import pathlib
import pickle
import tarfile

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error

def evaluate_model(name, model, X_test, y_test):
    print(f"Evaluating model: {name}...")
    y_pred = model.predict(X_test)

    # Get probability predictions if the model supports it
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_pred_prob = model.decision_function(X_test)
    else:
        # For XGBoost Booster model, get probability predictions using DMatrix
        print("last else")
        y_pred_prob = y_pred


    accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
    precision = precision_score(y_test, (y_pred > 0.5).astype(int))
    recall = recall_score(y_test, (y_pred > 0.5).astype(int))
    f1 = f1_score(y_test, (y_pred > 0.5).astype(int))
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f"Model {name} evaluation complete.")
    return {
        "regression_metrics": {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC": roc_auc
            # "mse": {"value": mse, "standard_deviation": std},
        }
    }

if __name__ == "__main__":
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    name='XGBoostModel'
    model = pickle.load(open("xgboost-model", "rb"))
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    # seperating target variable from feature data
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)

    # Convert to expecting format to run the model
    X_test = xgb.DMatrix(df.values)

    # creating metrics report from above
    evaluation_metrics = evaluate_model(name, model, X_test, y_test)

    # Creating and locally storing evaluation report
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(evaluation_metrics))
