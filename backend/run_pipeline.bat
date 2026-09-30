@echo off
echo Running ClimateAI India Data Pipeline...
echo.
echo Step 1: Preprocessing data...
python src/preprocessing.py
if %errorlevel% neq 0 goto error

echo.
echo Step 2: Creating features...
python src/features.py
if %errorlevel% neq 0 goto error

echo.
echo Step 3: Training temperature model...
python src/train_temperature.py
if %errorlevel% neq 0 goto error

echo.
echo Step 4: Training rainfall model...
python src/train_rainfall.py
if %errorlevel% neq 0 goto error

echo.
echo ========================================
echo Pipeline completed successfully!
echo ========================================
echo.
echo Now run: streamlit run app/app.py
pause
goto end

:error
echo.
echo ========================================
echo ERROR: Pipeline failed!
echo ========================================
pause

:end
