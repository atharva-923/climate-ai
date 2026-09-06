"""Train rainfall prediction model with class balancing and chronological split"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from pathlib import Path

def train_rainfall_model():
    """Train and save rainfall classification model with balanced class weights"""
    print("Training rainfall model...")

    # Load featured data
    df = pd.read_csv('data/processed/weather_featured.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    target_col = 'Next_Day_Rain_Category'

    if target_col not in df.columns:
        print(f"Error: {target_col} not found")
        return None

    # Feature columns matching temperature predictor
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
    y_raw_train = df.loc[train_mask, target_col]
    X_test = df.loc[test_mask, feature_cols]
    y_raw_test = df.loc[test_mask, target_col]

    # Fixed label ordering for logical categories
    ordered_classes = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain']
    le = LabelEncoder()
    le.fit(ordered_classes)

    y_train = le.transform(y_raw_train)
    y_test = le.transform(y_raw_test)

    # Compute balanced sample weights to address high frequency of 'No Rain'
    sample_weights = compute_sample_weight('balanced', y_train)

    print(f"Train size: {len(X_train):,} records, Test size: {len(X_test):,} records")
    print(f"Classes distribution in train:\n{y_raw_train.value_counts(normalize=True)}")

    # Train XGBoost multiclass classifier
    print("Fitting XGBoost Classifier with class weighting on 15-Year Master Dataset...")
    model = XGBClassifier(
        n_estimators=180,
        learning_rate=0.07,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_))).tolist()

    print(f"[OK] Rainfall Model Performance (Chronological Test Set 2023-2025):")
    print(f"  Accuracy:    {accuracy:.3f}")
    print(f"  Weighted F1: {f1_weighted:.3f}")
    print(f"  Macro F1:    {f1_macro:.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model, encoder, and metrics
    Path('models').mkdir(parents=True, exist_ok=True)
    joblib.dump(model, 'models/rainfall_model.pkl')
    joblib.dump(le, 'models/rainfall_encoder.pkl')

    metrics = {
        'accuracy': float(accuracy),
        'f1': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'features': feature_cols,
        'confusion_matrix': cm,
        'classes': le.classes_.tolist(),
        'train_records': int(len(X_train)),
        'test_records': int(len(X_test)),
        'split_strategy': 'Chronological (Train 2010-2022, Test 2023-2025 across 50 cities)'
    }
    joblib.dump(metrics, 'models/rainfall_metrics.pkl')

    print("[OK] Rainfall model, encoder, and metrics saved successfully.")
    return model, metrics

if __name__ == "__main__":
    train_rainfall_model()
