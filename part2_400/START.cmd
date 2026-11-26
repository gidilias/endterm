@echo off
cd /d "%~dp0"
python -m venv --system-site-packages .venv
call .venv\Scripts\activate.bat
python -c "import torch,transformers; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.cuda.get_device_name(0))"
if errorlevel 1 goto fail
python -m pip install -r requirements-extra.txt
if errorlevel 1 goto fail
python setup_official.py
if errorlevel 1 goto fail
python -c "import nltk; assert nltk.download('wordnet'); assert nltk.download('omw-1.4'); assert nltk.download('punkt'); assert nltk.download('punkt_tab')"
if errorlevel 1 goto fail
python run.py
if errorlevel 1 goto fail
python report.py --mlflow
if errorlevel 1 goto fail
start "" runs\dashboard.html
pause
exit /b 0
:fail
 echo STOPPED. Send the last error text. Existing results are preserved.
pause
