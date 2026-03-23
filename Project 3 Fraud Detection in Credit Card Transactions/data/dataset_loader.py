import kagglehub
import pandas as pd
import os


def download_and_load():
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = os.path.join(path, "creditcard.csv")
    df = pd.read_csv(csv_path)
    return df, path
