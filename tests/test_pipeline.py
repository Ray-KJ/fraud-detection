import os
import pandas as pd
import joblib
import xgboost as xgb

def test_processed_data_exists():
    """Check if the preprocessing script actually created the files."""
    assert os.path.exists("data/processed/X_test.csv"), "X_test.csv is missing!"
    assert os.path.exists("data/processed/y_test.csv"), "y_test.csv is missing!"

def test_feature_count():
    """Ensure the model always receives exactly 30 features."""
    df = pd.read_csv("data/processed/X_test.csv")
    # Our features: Time + V1-V28 + Amount = 30
    assert df.shape[1] == 30, f"Expected 30 features, but got {df.shape[1]}"

def test_model_loading():
    """Verify that the saved model is valid and can load."""
    model = xgb.XGBClassifier()
    try:
        model.load_model("models/fraud_model.json")
        loaded = True
    except:
        loaded = False
    assert loaded is True, "The XGBoost model file is corrupted or not in JSON format!"

def test_scaler_version():
    """Ensure the scaler was saved with the expected library."""
    scaler = joblib.load("models/scaler.pkl")
    # Verify it is a StandardScaler object
    from sklearn.preprocessing import StandardScaler
    assert isinstance(scaler, StandardScaler), "The scaler artifact is not a StandardScaler!"