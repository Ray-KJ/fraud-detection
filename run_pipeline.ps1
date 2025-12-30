$ErrorActionPreference = "Stop"

Write-Host "Starting Full Production Pipeline..." -ForegroundColor Cyan

# 1. ML Pipeline
Write-Host "Step 1: Ingesting Data..." -ForegroundColor Yellow
python src/ingest_data.py

Write-Host "Step 2: Preprocessing..." -ForegroundColor Yellow
python src/preprocess.py

Write-Host "Step 3: Training Model..." -ForegroundColor Yellow
python src/train.py

# 2. Docker Deployment
Write-Host "Step 4: Building Docker Image..." -ForegroundColor Blue
# We use --no-cache to ensure the new model and pinned versions are used
docker build --no-cache -t fraud-detection-api .

Write-Host "Step 5: Launching Container..." -ForegroundColor Blue
# This stops any old container running on port 8000 before starting the new one
docker stop fraud-detection-app 2>$null
docker rm fraud-detection-app 2>$null
docker run -d --name fraud-detection-app -p 8000:8000 fraud-detection-api

Write-Host "Pipeline Complete!" -ForegroundColor Green
Write-Host "API is live at http://localhost:8000/docs" -ForegroundColor Green