import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os


def load_data(path):
    csv_path = os.path.join(path, "creditcard.csv")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    return df


def preprocess(df):
    df = df.dropna()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    scaler = StandardScaler()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, "model/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test, scaler
