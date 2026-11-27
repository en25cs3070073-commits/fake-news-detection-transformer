@echo off
echo === Launching Fake News Detection Streamlit Demo ===
"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" %*
