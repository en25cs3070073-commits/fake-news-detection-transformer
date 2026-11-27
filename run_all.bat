@echo off
echo === Running Project 5: Fake News Detection Pipeline ===
set PY="%~dp0.venv\Scripts\python.exe"

echo [1/4] Preparing and cleaning dataset...
%PY% -m src.data_prep

echo [2/4] Training TF-IDF baselines...
%PY% -m src.train_baseline

echo [3/4] Fine-tuning DistilBERT Transformer...
%PY% -m src.train_transformer

echo [4/4] Building and executing Jupyter Notebook...
%PY% -m src.build_notebook
%PY% -m jupyter nbconvert --to notebook --execute --inplace "%~dp0notebooks\Fake_News_Detection_Transformer.ipynb"

echo === Pipeline Complete! Launch demo with run_demo.bat ===
pause
