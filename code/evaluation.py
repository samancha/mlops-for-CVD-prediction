import json
import pathlib
import pickle
import tarfile

import joblib
import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import mean_squared_error

def evaluate_model(name, model, X_test):
    print(f"Evaluating model: {name}...")
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f"Model {name} evaluation complete.")
    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC": roc_auc
    }

if __name__ == "__main__":
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    model = pickle.load(open("xgboost-model", "rb"))

    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    name= 'XGBoostMODEL'
    print("df shape", df.shape)
    print("df 0 ", df[0])
    evaluation_metrics = evaluate_model(name, model, df)

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(evaluation_metrics))

    # Code provided from lab example, add above proper code that outputs a file as model.tar.gz NO pickle
    # model_path = f"/opt/ml/processing/model/model.tar.gz"
    # with tarfile.open(model_path) as tar:
    #     tar.extractall(path=".")

    # model = pickle.load(open("xgboost-model", "rb"))

    # test_path = "/opt/ml/processing/test/test.csv"
    # df = pd.read_csv(test_path, header=None)

    # y_test = df.iloc[:, 0].to_numpy()
    # df.drop(df.columns[0], axis=1, inplace=True)

    # X_test = xgboost.DMatrix(df.values)

    # predictions = model.predict(X_test)

    # mse = mean_squared_error(y_test, predictions)
    # std = np.std(y_test - predictions)
    # report_dict = {
    #     "regression_metrics": {
    #         "mse": {"value": mse, "standard_deviation": std},
    #     },
    # }

    # output_dir = "/opt/ml/processing/evaluation"
    # pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # evaluation_path = f"{output_dir}/evaluation.json"
    # with open(evaluation_path, "w") as f:
    #     f.write(json.dumps(report_dict))
