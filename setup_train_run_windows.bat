@echo off
setlocal
if not exist .venv py -m venv .venv
call .venv\Scriptsctivate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m training.bootstrap_and_train
if errorlevel 1 (pause & exit /b 1)
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
