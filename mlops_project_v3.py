# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna
import joblib

# Load and preprocess the dataset
print("Loading the dataset...")
file_path = r'/Users/Steve/dev/GitProjects/mlops-for-CVD-prediction/cardio_train.csv'
df = pd.read_csv(file_path, sep=";")

print("Dataset loaded successfully. Now filling missing values with the median...")
df = df.fillna(df.median())  # Fill missing values

print("Converting 'gender' column to numeric (1=male, 2=female) as needed...")
df['gender'] = df['gender'].map({1: 0, 2: 1})  # Convert gender to numeric

print("Subsampling the dataset for faster training...")
df = df.sample(frac=0.2, random_state=42)  # Use 20% of the data for faster training

print("Splitting the data into features (X) and target (y)...")
X = df.drop(columns=['cardio'])  # Features
y = df['cardio']  # Target variable

print("Performing train-test split with 80% training and 20% testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train-test split completed. Training set size:", X_train.shape, ", Testing set size:", X_test.shape)

# Define a function to evaluate the model and return the metrics
def evaluate_model(name, model):
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

# Define Optuna objective function for hyperparameter tuning
def objective(trial, model_name):
    print(f"Starting hyperparameter tuning for {model_name} using Optuna...")
    if model_name == 'LogisticRegression':
        C = trial.suggest_float('C', 0.1, 1.0)  # Reduced range for faster tuning
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
        model = LogisticRegression(C=C, solver=solver, max_iter=100)  # Reduced max_iter for faster convergence
    elif model_name == 'GradientBoosting':
        learning_rate = trial.suggest_float('learning_rate', 0.05, 0.2)  # Narrowed range
        n_estimators = trial.suggest_int('n_estimators', 50, 100)  # Reduced n_estimators for faster training
        max_depth = trial.suggest_int('max_depth', 2, 4)  # Reduced max_depth for faster training
        model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    
    print(f"Training {model_name} with current trial parameters...")
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Finished trial with {model_name}, score: {score}")
    return score

# Function to tune, evaluate, and save models with Optuna
def tune_and_save_model(model_name, file_name):
    print(f"Tuning {model_name} with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_name), n_trials=5)  # Reduced n_trials for faster tuning

    print(f"Best hyperparameters found for {model_name}: {study.best_params}")
    
    # Instantiate the best model with the best parameters
    if model_name == 'LogisticRegression':
        model = LogisticRegression(**study.best_params, max_iter=100)
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(**study.best_params)
    
    print(f"Training {model_name} with the best found parameters...")
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print(f"Evaluating the tuned {model_name}...")
    evaluation_metrics = evaluate_model(model_name, model)
    print(f"Evaluation metrics for {model_name}: {evaluation_metrics}")
    
    # Save the model
    print(f"Saving the trained {model_name} model to {file_name}...")
    joblib.dump(model, file_name)
    print(f"Model {model_name} saved successfully.")
    
    return evaluation_metrics

# Tune, evaluate, and save Logistic Regression model
print("Starting the process for Logistic Regression model...")
logistic_metrics = tune_and_save_model('LogisticRegression', 'logistic_model_optuna.pkl')

# Tune, evaluate, and save Gradient Boosting model
print("Starting the process for Gradient Boosting model...")
gb_metrics = tune_and_save_model('GradientBoosting', 'gb_model_optuna.pkl')

# Combine the results into a DataFrame
print("Combining evaluation metrics from both models...")
performance_df = pd.DataFrame([logistic_metrics, gb_metrics])

print("Model Performance:")
print(performance_df)

# Save the performance metrics to a CSV file
performance_file = 'model_performance_logistic_gb.csv'
print(f"Saving model performance metrics to {performance_file}...")
performance_df.to_csv(performance_file, index=False)
print("Model performance metrics saved successfully.")
