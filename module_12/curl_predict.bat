@echo off
setlocal

curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"age\":34,\"annual_income\":82000,\"loan_amount\":18000,\"credit_score\":735,\"employment_years\":6,\"existing_debt\":12000}"

echo.
endlocal
