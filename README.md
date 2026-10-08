# Heart Disease Prediction with MLOps

A small end-to-end MLOps project for an academic demo.

## Pipeline

Data -> Train -> MLflow tracking -> Saved model -> FastAPI -> Docker
       |
       -> DVC versioning
       |
       -> Evidently drift report
       |
       -> GitHub Actions retraining/testing

The download step uses the UCI Cleveland Heart Disease dataset and converts
the original target values into a binary target (0 = no disease, 1 = disease).

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Download the dataset:

```powershell
python src\download_data.py
```

3. Train the model:

```powershell
python src\train.py
```

4. Open MLflow:

```powershell
mlflow ui --backend-store-uri "sqlite:///D:/PROJECTS/MLOPS HEART  DISEASE/heart_mlops_project/mlflow.db" --port 5000
```

Open http://127.0.0.1:5000

5. Run API:

```powershell
uvicorn api:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs

6. Create sample new data and drift report:

```powershell
python src\make_new_data.py
python src\monitor.py
```

Open `reports\drift_report.html` in a browser.

## DVC demo

This repository is initialized with DVC and uses `.\dvc_storage` as a local
remote. After changing the dataset, refresh its DVC pointer and push the data:

```powershell
dvc add data\heart.csv
git add .dvc .dvcignore .gitignore data\heart.csv.dvc data\.gitignore
git commit -m "Track heart disease data with DVC"
dvc push
```

The local remote is enough for a classroom demo on one computer. A shared
cloud remote is needed to share the data cache with other machines.

## Docker

Train first so the model files exist, then:

```powershell
docker build -t heart-mlops .
docker run -p 8000:8000 heart-mlops
```

Open http://127.0.0.1:8000/docs

## GitHub Actions

The workflow is already included at `.github/workflows/mlops.yml`. Push this repository to GitHub to run it on changes to the data or source files; it installs dependencies, trains, monitors drift, runs tests, and stores the generated artifacts. It can also be started manually from the Actions tab.
