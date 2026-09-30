"""Feature engineering for weather prediction with cyclical time and geospatial features"""
import pandas as pd
import numpy as np
from pathlib import Path

def create_features(df):
    """Create time-based, cyclical, geospatial, and lag features grouping by city"""
    print("Creating features...")

    # Ensure Date column is datetime
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    df['Date'] = pd.to_datetime(df[date_col])

    # Sort by city and date
    df = df.sort_values(['City', 'Date']).reset_index(drop=True)

    # Time & Cyclical features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Cyclical encoding for day of year and month (adjacent dates have close representations)
    df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)

    # Season mapping
    df['Season'] = df['Month'].apply(lambda x:
        'Winter' if x in [12, 1, 2] else
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else 'Autumn'
    )
    season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
    df['Season_Encoded'] = df['Season'].map(season_map).fillna(0).astype(int)

    # Identifiers
    temp_col = 'Temperature'
    rain_col = 'Rainfall'
    wind_col = 'WindSpeed'

    # Group by city for shift and rolling operations to prevent cross-city leakage
    grouped = df.groupby('City')

    # Lag features for temperature
    df['Temp_Lag1'] = grouped[temp_col].shift(1)
    df['Temp_Lag3'] = grouped[temp_col].shift(3)
    df['Temp_Lag7'] = grouped[temp_col].shift(7)

    # Rolling features for temperature
    df['Temp_Roll3'] = grouped[temp_col].rolling(window=3).mean().reset_index(0, drop=True)
    df['Temp_Roll7'] = grouped[temp_col].rolling(window=7).mean().reset_index(0, drop=True)

    # Lag features for rainfall & wind
    df['Rain_Lag1'] = grouped[rain_col].shift(1)
    df['Rain_Lag3'] = grouped[rain_col].shift(3)
    df['Wind_Lag1'] = grouped[wind_col].shift(1)

    # Current rainfall category
    df['Rain_Category'] = pd.cut(
        df[rain_col],
        bins=[-np.inf, 0.1, 2.5, 10.0, np.inf],
        labels=['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain']
    )

    # Target variable: next day temperature
    df['Next_Day_Temp'] = grouped[temp_col].shift(-1)

    # Target variable: next day rainfall category
    next_day_rain = grouped[rain_col].shift(-1)
    df['Next_Day_Rain_Category'] = pd.cut(
        next_day_rain,
        bins=[-np.inf, 0.1, 2.5, 10.0, np.inf],
        labels=['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain']
    )

    # Drop NaN rows created by lag (initial days) and lead (last day) features
    df = df.dropna().reset_index(drop=True)

    print(f"[OK] Features created: {len(df)} rows across {df['City'].nunique()} cities")
    return df

if __name__ == "__main__":
    processed_path = Path('data/processed/weather_processed.csv')
    if not processed_path.exists():
        from preprocessing import load_and_clean_data
        load_and_clean_data()
    df = pd.read_csv(processed_path)
    df = create_features(df)
    output_path = Path('data/processed/weather_featured.csv')
    df.to_csv(output_path, index=False)
    print(f"[OK] Featured data saved to {output_path}")
