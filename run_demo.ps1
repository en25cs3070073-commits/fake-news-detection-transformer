Write-Host "=== Launching Fake News Detection Streamlit Demo ===" -ForegroundColor Cyan
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m streamlit run (Join-Path $PSScriptRoot "app.py")
