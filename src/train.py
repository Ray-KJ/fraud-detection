import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import classification_report, average_precision_score

def run_training(data_dir, model_dir):
    print("Loading processed data...")
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()

    print("Training XGBoost model...")
    # Scale_pos_weight is a common alternative to SMOTE, 
    # but since we already used SMOTE, we'll use standard parameters.
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # 1. Evaluation
    print("valuating model...")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # We focus on AUPRC because the dataset is imbalanced
    auprc = average_precision_score(y_test, y_probs)
    
    print(f"Model AUPRC: {auprc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 2. Save the model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "fraud_model.pkl")
    model.save_model("models/fraud_model.json")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    run_training('data/processed', 'models')