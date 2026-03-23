import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def run_isolation_forest(X_train, X_test):
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    train_scores = iso.decision_function(X_train)
    test_scores = iso.decision_function(X_test)
    return train_scores, test_scores


def run_lof(X_train, X_test):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True, n_jobs=-1)
    lof.fit(X_train)
    train_scores = lof.decision_function(X_train)
    test_scores = lof.decision_function(X_test)
    return train_scores, test_scores


def add_anomaly_features(X_train, X_test):
    import pandas as pd

    iso_train, iso_test = run_isolation_forest(X_train, X_test)
    lof_train, lof_test = run_lof(X_train, X_test)

    X_train = pd.DataFrame(X_train).copy()
    X_test = pd.DataFrame(X_test).copy()

    X_train["iso_score"] = iso_train
    X_train["lof_score"] = lof_train
    X_test["iso_score"] = iso_test
    X_test["lof_score"] = lof_test

    return X_train, X_test
