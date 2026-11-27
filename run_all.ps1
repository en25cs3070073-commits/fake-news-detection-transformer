Write-Host "=== Running Project 5: Fake News Detection Pipeline ===" -ForegroundColor Cyan
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "[1/4] Preparing & cleaning dataset..." -ForegroundColor Yellow
& $py -m src.data_prep

Write-Host "[2/4] Training TF-IDF baselines..." -ForegroundColor Yellow
& $py -m src.train_baseline

Write-Host "[3/4] Fine-tuning DistilBERT Transformer..." -ForegroundColor Yellow
& $py -m src.train_transformer

Write-Host "[4/4] Building & executing Jupyter Notebook..." -ForegroundColor Yellow
& $py -m src.build_notebook
& $py -m jupyter nbconvert --to notebook --execute --inplace (Join-Path $PSScriptRoot "notebooks\Fake_News_Detection_Transformer.ipynb")

Write-Host "=== Pipeline Complete! Launch demo with .\run_demo.ps1 ===" -ForegroundColor Green
