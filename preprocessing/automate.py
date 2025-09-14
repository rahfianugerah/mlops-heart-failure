#!/usr/bin/env python3

import os
import shutil
import subprocess
import pandas as pd

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

RAW_CSV       = "heart_failure_raw.csv"
OUT_TRAIN     = "preprocessing/preprocess_dataset/train_preprocessed.csv"
OUT_TEST      = "preprocessing/preprocess_dataset/test_preprocessed.csv"
PKL_DIR       = "preprocessing/preprocess_data_pkl"

TARGET_COL    = "HeartDisease"
NUM_FEATURES  = ["Age","RestingBP","Cholesterol","MaxHR","Oldpeak"]
CAT_FEATURES  = ["Sex","ChestPainType","FastingBS","RestingECG","ExerciseAngina","ST_Slope"]

TEST_SIZE     = 0.2
RANDOM_STATE  = 42

KAGGLE_DATASET = "fedesoriano/heart-failure-prediction"
KAGGLE_FILE    = "heart.csv"

def download_dataset(output_path: str = RAW_CSV, dataset: str = KAGGLE_DATASET) -> None:
    if os.path.exists(output_path):
        return

    if shutil.which("kaggle") is None:
        print("[WARN] Kaggle CLI Not Found - Skip Download")
        return

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-f", KAGGLE_FILE, "-p", ".", "--unzip"],
            check=True
        )
        if os.path.exists(KAGGLE_FILE):
            os.replace(KAGGLE_FILE, output_path)
            print(f"[INFO] Downloaded Dataset: {output_path}")
        else:
            print("[WARN] Downloaded - File Not Found After Unzip.")
    except Exception as e:
        print(f"[WARN] Failed to Download via Kaggle: {e}")

def preprocess(
    input_csv: str,
    target_col: str,
    num_features: List[str],
    cat_features: List[str],
    test_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if not os.path.exists(input_csv):
        download_dataset(output_path=input_csv, dataset=KAGGLE_DATASET)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Raw CSV Not Found: {input_csv}")

    df = pd.read_csv(input_csv)

    df.columns   = [c.lower() for c in df.columns]
    target_col   = target_col.lower()
    num_features = [c.lower() for c in num_features]
    cat_features = [c.lower() for c in cat_features]

    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip().str.lower()

    for col in cat_features:
        if col not in df.columns:
            raise ValueError(f"Missing Categorical Column: {col}")
        df[col] = LabelEncoder().fit_transform(df[col].astype("string"))

    if target_col not in df.columns:
        raise ValueError(f"Missing Target Column: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if num_features:
        scaler = StandardScaler()
        X_train[num_features] = scaler.fit_transform(X_train[num_features])
        X_test[num_features]  = scaler.transform(X_test[num_features])

    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df  = pd.concat([X_test.reset_index(drop=True),  y_test.reset_index(drop=True)], axis=1)
    return train_df, test_df

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TEST), exist_ok=True)
    os.makedirs(PKL_DIR, exist_ok=True)

    train_df, test_df = preprocess(
        RAW_CSV, TARGET_COL, NUM_FEATURES, CAT_FEATURES, TEST_SIZE, RANDOM_STATE
    )

    train_df.to_csv(OUT_TRAIN, index=False)
    test_df.to_csv(OUT_TEST, index=False)

    train_df.to_pickle(os.path.join(PKL_DIR, "train_preprocessed.pkl"))
    test_df.to_pickle(os.path.join(PKL_DIR, "test_preprocessed.pkl"))

    print("[OK] Preprocessing Completed")
    print(f" -> {OUT_TRAIN}")
    print(f" -> {OUT_TEST}")
