import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

def run_basic_modelling():
    print("=== STARTING MLFLOW BASIC MODELLING WITH AUTOLOG ===")
    
    # Membaca parameter path data yang dikirim oleh MLproject secara otomatis
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="namadataset_preprocessing/dataset_clean.csv")
    args = parser.parse_args()
    
    data_path = args.data_path
    print(f"[INFO] Membaca data dari jalur: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Data tidak ditemukan di: {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Memisahkan fitur dan target
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Menentukan jenis model secara cerdas berdasarkan tipe data target
    with mlflow.start_run(run_name="Baseline_Model_Autolog"):
        if y_train.dtype == 'object' or len(np.unique(y_train)) < 10:
            print("[INFO] Target berupa kategori/klasifikasi. Menggunakan RandomForestClassifier.")
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            print(f"[SUCCESS] Baseline Accuracy Score: {accuracy_score(y_test, preds):.4f}")
        else:
            print("[INFO] Target berupa angka kontinu. Menggunakan RandomForestRegressor.")
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            print(f"[SUCCESS] Baseline R2 Score: {r2_score(y_test, preds):.4f}")

if __name__ == "__main__":
    run_basic_modelling()