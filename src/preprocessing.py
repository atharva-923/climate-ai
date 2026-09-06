"""Data preprocessing for ClimateAI India - 15-Year Unified Dataset (2010–2025)"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_and_clean_data():
    """Load both 2010-2019 and 2020-2025 raw datasets and merge into a unified 15-year dataset"""
    print("Loading datasets for 15-year unification (2010–2025)...")

    # Load cities
    cities_df = pd.read_csv('data/raw/cities.csv')
    cities_df = cities_df.rename(columns={'city_name': 'City'})
    print(f"[OK] Loaded {len(cities_df)} cities")

    # 1. Load 2010–2019 Decadal Dataset
    df_2010_2019 = pd.read_csv('data/raw/daily_data_combined_2010_to_2019.csv')
    print(f"[OK] Loaded 2010–2019 dataset: {len(df_2010_2019):,} records")

    # 2. Load 2020–2025 Modern Dataset (if available)
    raw_path_2020 = Path('data/raw/daily_data.csv')
    if raw_path_2020.exists():
        df_2020_2025 = pd.read_csv(raw_path_2020)
        print(f"[OK] Loaded 2020–2025 dataset: {len(df_2020_2025):,} records")
        weather_df = pd.concat([df_2010_2019, df_2020_2025], ignore_index=True)
    else:
        weather_df = df_2010_2019.copy()

    print(f"[OK] Combined raw records: {len(weather_df):,}")

    # Standardize column names
    weather_df = weather_df.rename(columns={
        'city_name': 'City',
        'datetime': 'Date',
        'temperature_2m_mean': 'Temperature',
        'precipitation_sum': 'Rainfall',
        'wind_speed_10m_max': 'WindSpeed'
    })

    # Convert date
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])

    # Select key columns
    key_cols = ['City', 'Date', 'Temperature', 'Rainfall', 'WindSpeed']
    weather_df = weather_df[key_cols].copy()

    # Drop duplicate city + date observations
    weather_df = weather_df.drop_duplicates(subset=['City', 'Date'])

    # Sort chronologically by City and Date
    weather_df = weather_df.sort_values(['City', 'Date']).reset_index(drop=True)

    # Merge with city coordinates
    weather_df = weather_df.merge(
        cities_df[['City', 'latitude', 'longitude']],
        on='City',
        how='left'
    )

    # Remove rows with missing key values
    print(f"Records before cleaning: {len(weather_df):,}")
    weather_df = weather_df.dropna(subset=['Temperature', 'Rainfall', 'WindSpeed'])
    print(f"Records after cleaning: {len(weather_df):,}")

    # Save processed data
    output_path = Path('data/processed/weather_processed.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(output_path, index=False)

    print(f"[OK] 15-Year Unified dataset saved to {output_path}")
    print(f"[OK] Cities: {weather_df['City'].nunique()}")
    print(f"[OK] Date range: {weather_df['Date'].min().strftime('%Y-%m-%d')} to {weather_df['Date'].max().strftime('%Y-%m-%d')}")
    return weather_df

if __name__ == "__main__":
    load_and_clean_data()
