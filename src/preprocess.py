import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def run_preprocessing(input_path, output_dir):
    print("Starting preprocessing...")
    df = pd.read_csv(input_path)
    
    # 1. Initialize and fit the Scaler
    # We only scale 'Time' and 'Amount'
    scaler = StandardScaler()
    
    # We fit on the data to learn the mean and standard deviation
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    
    # 2. Save the scaler for production use!
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    # 3. Features and Target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 4. Train/Test Split (80/20)
    # We split BEFORE SMOTE to ensure the test set remains realistic
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Handle Imbalance with SMOTE (Training set only)
    print("Applying SMOTE to balance classes...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    # 6. Save processed data for the training script
    os.makedirs(output_dir, exist_ok=True)
    X_train_res.to_csv(f"{output_dir}/X_train.csv", index=False)
    y_train_res.to_csv(f"{output_dir}/y_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    print(f"Preprocessed data saved to {output_dir}")

if __name__ == "__main__":
    run_preprocessing('data/raw/creditcard.csv', 'data/processed')