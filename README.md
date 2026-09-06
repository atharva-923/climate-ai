# 🌦️ ClimateAI India

**Next-Generation Atmospheric Intelligence, Decadal Climate Analytics & Aerospace Turbulence Telemetry Platform**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EB6434.svg?logo=xgboost&logoColor=white)](https://xgboost.ai)
[![Open-Meteo](https://img.shields.io/badge/Live_Telemetry-Open--Meteo-38BDF8.svg)](https://open-meteo.com)
[![Status](https://img.shields.io/badge/Status-Hackathon_Ready-success.svg)]()

ClimateAI India is an end-to-end meteorological and aerospace intelligence platform designed specifically for 50 major Indian metropolitan hubs and climate zones. It bridges **10+ years of decadal historical data** (182,600+ verified daily weather observations from 2010–2019) with **real-time live sensor telemetry**, **10-day multi-horizon AI forecasting**, and **aviation-grade atmospheric turbulence hazard modeling**.

---

## 🌟 Key Highlights & Innovations

### 1. 📡 Live Atmospheric & Sensor Telemetry Stream
- **50-Station Live Telemetry:** Real-time atmospheric streaming across 50 Indian cities via Open-Meteo API.
- **Air Quality Telemetry (AQI):** Live monitoring of PM2.5, PM10, surface ozone, with automated health advisory ratings.
- **Live Data Ticker:** Real-time auto-scrolling telemetry tape displaying nationwide temperature, humidity, and weather conditions.
- **Automated Weather Alerts:** Real-time threshold detection for heatwaves, severe downpours, high winds, and severe thermal departure.

### 2. 🔮 Multi-Horizon AI Forecasting & Verification
- **Dual ML Engine:**
  - **Temperature Regressor (XGBoost):** Highly tuned regressor predicting next-day ambient temperature with **0.68°C – 0.86°C MAE** and **>93% R²**.
  - **Rainfall Classifier (XGBoost):** 4-tier probabilistic classification (`No Rain`, `Light Rain`, `Moderate Rain`, `Heavy Rain`) with **73.3% accuracy** and **72.0% weighted F1-score**.
- **10-Day Multi-Horizon Outlook:** 10-day forecasts with daily minimum/maximum temperatures, condition envelopes, and precipitation likelihoods.
- **24-Hour Hourly Progression:** Granular dual-axis temperature curve and hourly rain probability bars.
- **15-Day Historical Verification Trajectory:** Interactive validation window comparing **Observed Ground Truth** vs. **AI Prediction** vs. **10-Year Climatological Baseline**.
- **Climatological Event Presets:** Instant 1-click loading of historic meteorological extremes (e.g., 2024 Record Heatwave, 2023 Monsoon Inundation, 2018 Winter Chill).

### 3. ✈️ Aerospace & Aviation Meteorology Hub
- **Clear-Air Turbulence (CAT) Severity Index:** Dynamic index (0–100) computing low-altitude and cruise turbulence risk.
- **Density Altitude & Pressure Altitude:** Accurate aerodrome density altitude calculations and ISA temperature departures.
- **Boundary-Layer Wind Shear:** Real-time vertical shear gradients in knots per 1,000 ft (`kt/kft`).
- **Takeoff Roll Penalty & Icing Risk:** Aerodynamic degradation indices and FL180 structural icing probabilities.
- **Interactive Flight Corridor Cross-Section:** Route cross-section simulations across major Indian air corridors:
  - `DEL ✈ BOM` (Delhi ↔ Mumbai)
  - `BLR ✈ DEL` (Bangalore ↔ Delhi)
  - `BOM ✈ GOI` (Mumbai ↔ Goa)
  - `MAA ✈ CCU` (Chennai ↔ Kolkata)
  - `DEL ✈ SXR` (Delhi ↔ Srinagar Himalayan Wave Corridor)
  - `HYD ✈ BOM` (Hyderabad ↔ Mumbai)

### 4. 🗺️ Geospatial Doppler Radar Map
- Dark-matter GIS map plotting all 50 active meteorological stations.
- Multi-layer toggle:
  - Live Temperature Field & Heatmap
  - Live Precipitation & Radar Density
  - Aviation CAT Turbulence Risk
  - Inter-City Flight Corridor Vector Lines
- Detailed station popup cards displaying micro-climate telemetry.

### 5. 🏙️ Station Location Analytics
- **Thermal Climatology:** Decadal temperature progression, seasonal spline trajectories, and 10-year monthly thermal heatmaps.
- **Precipitation Dynamics:** Annual monsoon analysis, dry vs. wet season ratios, and precipitation distributions.
- **Comparative Station Benchmark:** Side-by-side comparison of any two Indian cities across key climatological indicators.

### 6. 🧪 Scenario Simulator (What-If Climate Stress-Testing)
- **2D ML Iso-Contour Sensitivity Heatmap:** Visualizes the model's non-linear response surface in (Temperature × Rainfall) parameter space.
- **Operating Crosshair Tracking:** Real-time marker tracking user adjustments across the decision plane.
- **Comparative Outcome Bars:** Instant delta evaluation of baseline vs. simulated outcome for temperature and rainfall category.

### 7. 📈 Transparent Model Benchmarks
- Rigorous temporal train/validation/test split (Train: 2010–2017, Val: 2018, Test: 2019; 36,450 held-out test points).
- Feature importance rankings (Gini importance) for both models.
- Transparent reporting of MAE, RMSE, R², Multi-class Accuracy, and Weighted F1.

---

## 🏛️ System Architecture

```
climate-ai/
│
├── app/
│   └── app.py                      # Production Streamlit platform (2,600+ lines, Obsidian Glass UI)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Cleans raw meteorological records & handles missing data
│   ├── features.py                 # Cyclical temporal features, lag variables, rolling statistics
│   ├── live_weather.py             # Open-Meteo live API, AQI engine, batch fetch, aerospace telemetry
│   ├── train_temperature.py        # XGBoost temperature regression pipeline & model serialization
│   └── train_rainfall.py           # XGBoost rainfall multi-class classifier pipeline
│
├── models/
│   ├── temperature_model.pkl       # Serialized XGBoost temperature regressor
│   ├── temperature_metrics.pkl     # Validation & test metrics for temperature
│   ├── rainfall_model.pkl          # Serialized XGBoost rainfall classifier
│   ├── rainfall_encoder.pkl        # Label encoder for rainfall classes
│   └── rainfall_metrics.pkl        # Validation & test metrics for rainfall
│
├── data/
│   ├── raw/                        # Historical CSV datasets (50 Indian cities, 2010–2019)
│   └── processed/                  # Processed training and featured datasets
│
├── requirements.txt                # Python package dependencies
├── launch_app.bat                  # One-click Windows application launcher
├── install.bat                     # Dependency installer script
└── README.md
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed.
- Git installed.

### 1. Clone the Repository
```bash
git clone https://github.com/atharva-923/climate-ai.git
cd climate-ai
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
**Option A: One-click launcher (Windows)**
```cmd
launch_app.bat
```

**Option B: Via Command Line**
```bash
streamlit run app/app.py
```

The application will launch locally at **http://localhost:8501** (or port `8502`).

---

## ⚙️ Model Retraining & Pipeline Execution

To rebuild processed datasets and retrain the machine learning models from scratch:

```bash
# 1. Clean and preprocess raw data
python src/preprocessing.py

# 2. Extract cyclical, rolling, and lag features
python src/features.py

# 3. Train the XGBoost Temperature Regressor
python src/train_temperature.py

# 4. Train the XGBoost Rainfall Classifier
python src/train_rainfall.py
```

---

## 📊 Benchmark & Performance Summary

| Model | Task | Algorithm | Primary Metric | Baseline Score |
| :--- | :--- | :--- | :--- | :--- |
| **Temperature** | Continuous Forecasting | XGBoost Regressor | **MAE** | **0.68°C – 0.86°C** |
| | | | **R² Score** | **92.7% – 97.5%** |
| | | | **RMSE** | **1.79°C** |
| **Rainfall** | 4-Tier Classification | XGBoost Classifier | **Accuracy** | **73.3%** |
| | | | **Weighted F1** | **0.720** |

*Validation Scheme: Out-of-time temporal validation on unseen test data.*

---

## 🛠️ Technology Stack

- **Application Framework:** Streamlit
- **Machine Learning:** XGBoost, Scikit-learn, Joblib
- **Telemetry & Live APIs:** Open-Meteo Weather & Air Quality API, Requests
- **Data Engineering:** Pandas, NumPy
- **Interactive Visualizations:** Plotly (Express & Graph Objects), Folium, Streamlit-Folium
- **UI Design System:** Obsidian Glassmorphism with dynamic Aurora CSS animation engine

---

## 📜 Disclaimer
This software is developed for hackathon demonstration, educational, and research purposes. Do not use as a primary source for critical navigation or official aeronautical dispatch.

---

Built with ❤️ for Indian meteorological intelligence and climate resilience.
