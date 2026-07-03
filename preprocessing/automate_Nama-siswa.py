import os
import pandas as pd
import numpy as np

def load_data(file_path):
    """Fungsi untuk membaca dataset mentah"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {file_path}")
    print(f"[1/4] Membaca data dari {file_path}...")
    return pd.read_csv(file_path)

def clean_and_preprocess(df):
    """Fungsi pusat untuk semua logik preprocessing Anda"""
    print("[2/4] Menjalankan proses pembersihan data...")

    # 1. Salin dataframe agar tidak merubah data asli
    df_clean = df.copy()

    # 2. Handling Missing Values (Contoh: Mengisi dengan Mean/Modus atau Drop)
    # Jalankan logika yang sudah Anda temukan dari hasil eksperimen di Notebook
    kolom_numerik = df_clean.select_dtypes(include=[np.number]).columns
    for col in kolom_numerik:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    # 3. Handling Duplicates
    if df_clean.duplicated().sum() > 0:
        df_clean.drop_duplicates(inplace=True)
        print(f"     -> Duplikat data berhasil dihapus.")

    return df_clean

def save_data(df, output_folder, filename):
    """Fungsi untuk membuat folder baru dan menyimpan data hasil preprocessing"""
    print(f"[3/4] Menyiapkan folder output di: {output_folder}")
    # Membuat folder otomatis jika belum ada
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, filename)
    df.to_csv(output_path, index=False)
    print(f"[4/4] Data sukses diproses & disimpan di: {output_path}")

if __name__ == "__main__":
    # --- KONFIGURASI PATH UTK LOKAL ---
    INPUT_FILE = "namadataset_raw/Dataset Pelatihan.csv" 
    OUTPUT_FOLDER = "preprocessing/namadataset_preprocessing"
    OUTPUT_FILE = "dataset_clean.csv"
    
    print("=== MEMULAI AUTOMATISASI PREPROCESSING ===")
    # ... (sisa kode di bawahnya biarkan tetap sama)

    print("=== MEMULAI AUTOMATISASI PREPROCESSING ===")
    try:
        # Eksekusi Pipeline
        raw_data = load_data(INPUT_FILE)
        cleaned_data = clean_and_preprocess(raw_data)
        save_data(cleaned_data, OUTPUT_FOLDER, OUTPUT_FILE)
        print("=== PROSES SELESAI DENGAN SUKSES ===")
    except Exception as e:
        print(f"\n[ERROR] Terjadi kesalahan: {e}")
