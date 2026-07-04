import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

def run_tuning_modelling():
    print("=== STARTING MLFLOW TUNING WITH MANUAL LOGGING ===")
    
    data_path = "namadataset_preprocessing/dataset_clean.csv"
    df = pd.read_csv(data_path)
    
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    n_estimators = 120
    max_depth = 10
    
    with mlflow.start_run(run_name="Tuned_Model_Manual_Log"):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        if y_train.dtype == 'object' or len(np.unique(y_train)) < 10:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"[SUCCESS] Tuned Model -> Accuracy: {acc:.4f}")
            mlflow.log_metric("accuracy", acc)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            print(f"[SUCCESS] Tuned Model -> R2: {r2:.4f}, MSE: {mse:.4f}")
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)
            
        mlflow.sklearn.log_model(model, "random_forest_tuned_model")

if __name__ == "__main__":
    run_tuning_modelling()