import kagglehub
import shutil
import os

def download_data():
    print("Starting data ingestion from Kaggle...")
    
    # Downloads to a hidden cache folder by default
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    
    # We want it in our local project 'data' folder for the pipeline to see
    os.makedirs('data/raw', exist_ok=True)
    
    # Locate the .csv file and move it to our project
    for file in os.listdir(path):
        if file.endswith(".csv"):
            shutil.copy(os.path.join(path, file), 'data/raw/creditcard.csv')
            print(f"Data saved to data/raw/creditcard.csv")

if __name__ == "__main__":
    download_data()