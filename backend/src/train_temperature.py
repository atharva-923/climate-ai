"""Train temperature prediction model with leak-free chronological split and enhanced features"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

def train_temperature_model():
    """Train and save temperature model with chronological train/test split"""
    print("Training temperature model...")

    # Load featured data
    df = pd.read_csv('data/processed/weather_featured.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    target_col = 'Next_Day_Temp'

    # Select comprehensive feature set
    feature_cols = [
        'Temperature', 'Rainfall', 'WindSpeed',
        'Temp_Lag1', 'Temp_Lag3', 'Temp_Lag7',
        'Temp_Roll3', 'Temp_Roll7',
        'Rain_Lag1', 'Rain_Lag3', 'Wind_Lag1',
        'latitude', 'longitude',
        'Month', 'DayOfYear',
        'DayOfYear_Sin', 'DayOfYear_Cos',
        'Month_Sin', 'Month_Cos',
        'Season_Encoded'
    ]

    # Chronological Split (Train: 2010-2022, Test: 2023-2025) across ALL 50 cities
    split_date = pd.to_datetime('2023-01-01')
    train_mask = df['Date'] < split_date
    test_mask = df['Date'] >= split_date

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, target_col]

    print(f"Train size: {len(X_train):,} records ({df.loc[train_mask, 'Date'].min().date()} to {df.loc[train_mask, 'Date'].max().date()})")
    print(f"Test size:  {len(X_test):,} records ({df.loc[test_mask, 'Date'].min().date()} to {df.loc[test_mask, 'Date'].max().date()})")

    # Train XGBoost regressor
    print("Fitting XGBoost Regressor on 15-Year Master Dataset...")
    model = XGBRegressor(
        n_estimators=180,
        learning_rate=0.07,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate on unseen future test set (2023-2025)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"[OK] Temperature Model Performance (Chronological Test Set 2023-2025):")
    print(f"  MAE:  {mae:.3f}°C")
    print(f"  RMSE: {rmse:.3f}°C")
    print(f"  R²:   {r2:.4f}")

    # Save model and metrics
    Path('models').mkdir(parents=True, exist_ok=True)
    joblib.dump(model, 'models/temperature_model.pkl')

    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'features': feature_cols,
        'train_records': int(len(X_train)),
        'test_records': int(len(X_test)),
        'split_strategy': 'Chronological (Train 2010-2022, Test 2023-2025 across 50 cities)'
    }
    joblib.dump(metrics, 'models/temperature_metrics.pkl')

    print("[OK] Temperature model and metrics saved successfully.")
    return model, metrics

if __name__ == "__main__":
    train_temperature_model()
