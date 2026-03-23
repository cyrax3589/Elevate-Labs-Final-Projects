import kagglehub
import joblib
import os
from xgboost import XGBClassifier

from preprocess import load_data, preprocess
from anomaly_detection import add_anomaly_features
from utils import evaluate_model, save_roc_curve, save_confusion_matrix


def main():
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

    df = load_data(path)

    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    X_train, X_test = add_anomaly_features(X_train, X_test)

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred, y_prob = evaluate_model(model, X_test, y_test)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/xgb_model.pkl")

    save_roc_curve(y_test, y_prob)
    save_confusion_matrix(y_test, y_pred)

    print("Model and plots saved.")


if __name__ == "__main__":
    main()
