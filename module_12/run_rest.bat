@echo off
setlocal

python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

endlocal
