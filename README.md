<div align="center">

# 🌦️ ClimateAI India

### Next-Generation Atmospheric Intelligence & Aerospace Telemetry Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EB6434?style=for-the-badge)](https://xgboost.ai)
[![Open-Meteo](https://img.shields.io/badge/API-Open--Meteo-38BDF8?style=for-the-badge)](https://open-meteo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)]()

*A production-grade Streamlit meteorological platform fusing 10+ years of historical climate records
with real-time live sensor telemetry, 10-day AI forecasting, and aviation-grade turbulence hazard
modeling — across 50 major Indian cities.*

</div>

---

## 📸 Preview

The platform features an **Obsidian Glassmorphism UI** with a live aurora animation engine,
real-time telemetry ticker, and interactive Doppler-style radar maps.

*Run the app locally to experience it — see [Quick Start](#-quick-start) below.*

---

## ✨ Features

<details>
<summary><b>📡 Live Atmospheric & Sensor Telemetry Stream</b></summary>

- **50-Station Live Telemetry** — Real-time atmospheric streaming across 50 Indian cities via [Open-Meteo API](https://open-meteo.com) (no API key required).
- **Air Quality Index (AQI)** — Live monitoring of PM2.5, PM10, and surface ozone with automated health advisories.
- **Live Data Ticker** — Auto-scrolling telemetry tape displaying real-time temperature, humidity, and weather conditions nationwide.
- **Automated Alerts** — Real-time threshold detection for heatwaves, severe downpours, high winds, and extreme thermal departures.

</details>

<details>
<summary><b>🔮 Multi-Horizon AI Forecasting</b></summary>

- **Dual ML Engine:**
  - **Temperature Regressor (XGBoost):** Predicts next-day ambient temperature with **0.68°C–0.86°C MAE** and **>93% R²**.
  - **Rainfall Classifier (XGBoost):** 4-tier probabilistic classification (`No Rain`, `Light Rain`, `Moderate Rain`, `Heavy Rain`) achieving **73.3% accuracy** and **72.0% weighted F1**.
- **10-Day Multi-Horizon Outlook** — Daily min/max temperatures, condition envelopes, and precipitation likelihoods.
- **24-Hour Hourly Progression** — Granular dual-axis temperature curve with hourly rain probability bars.
- **15-Day Historical Verification** — Compares **Observed Ground Truth** vs. **AI Prediction** vs. **10-Year Climatological Baseline**.
- **Climatological Event Presets** — 1-click loading of historic extremes (2024 Heatwave, 2023 Monsoon Inundation, 2018 Winter Chill).

</details>

<details>
<summary><b>✈️ Aerospace & Aviation Meteorology Hub</b></summary>

- **Clear-Air Turbulence (CAT) Severity Index** — Dynamic 0–100 index for low-altitude and cruise turbulence risk.
- **Density Altitude & Pressure Altitude** — Accurate aerodrome calculations and ISA temperature departures.
- **Boundary-Layer Wind Shear** — Real-time vertical shear gradients in `kt/kft`.
- **Takeoff Roll Penalty & Icing Risk** — Aerodynamic degradation indices and FL180 structural icing probabilities.
- **Interactive Flight Corridor Cross-Sections** across 6 major Indian air corridors:

| Route | Corridor |
|:------|:---------|
| `DEL ✈ BOM` | Delhi ↔ Mumbai |
| `BLR ✈ DEL` | Bangalore ↔ Delhi |
| `BOM ✈ GOI` | Mumbai ↔ Goa |
| `MAA ✈ CCU` | Chennai ↔ Kolkata |
| `DEL ✈ SXR` | Delhi ↔ Srinagar (Himalayan Wave Corridor) |
| `HYD ✈ BOM` | Hyderabad ↔ Mumbai |

</details>

<details>
<summary><b>🗺️ Geospatial Doppler Radar Map</b></summary>

- Dark-matter GIS map plotting all 50 active meteorological stations.
- Multi-layer toggles: Live Temperature Heatmap · Precipitation Radar · Aviation CAT Risk · Flight Corridor Vectors.
- Detailed station popup cards with micro-climate telemetry.

</details>

<details>
<summary><b>🏙️ Station Location Analytics</b></summary>

- **Thermal Climatology** — Decadal temperature progression, seasonal splines, and 10-year monthly heatmaps.
- **Precipitation Dynamics** — Annual monsoon analysis, dry/wet season ratios, and rainfall distributions.
- **Comparative Station Benchmark** — Side-by-side comparison of any two Indian cities across key climatological indicators.

</details>

<details>
<summary><b>🧪 Scenario Simulator (What-If Climate Stress Testing)</b></summary>

- **2D ML Iso-Contour Sensitivity Heatmap** — Visualizes the model's non-linear response surface in (Temperature × Rainfall) parameter space.
- **Operating Crosshair Tracking** — Real-time marker tracking user adjustments across the decision plane.
- **Comparative Outcome Bars** — Instant delta evaluation of baseline vs. simulated outcome.

</details>

<details>
<summary><b>📈 Transparent Model Benchmarks</b></summary>

- Rigorous temporal train/validation/test split — Train: 2010–2022, Test: 2023–2025.
- Feature importance rankings (Gini importance) for both models.
- Full reporting of MAE, RMSE, R², Multi-class Accuracy, and Weighted F1.

</details>

---

## 🏛️ Architecture

```
climate-ai/
│
├── app/
│   └── app.py                      # Streamlit platform (~2,600 lines, Obsidian Glass UI)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Cleans raw records & handles missing data
│   ├── features.py                 # Cyclical temporal features, lag vars, rolling stats
│   ├── live_weather.py             # Open-Meteo API, AQI engine, batch fetch, aerospace telemetry
│   ├── train_temperature.py        # XGBoost temperature regression pipeline
│   └── train_rainfall.py           # XGBoost rainfall multi-class classifier pipeline
│
├── models/
│   ├── temperature_model.pkl       # Serialized XGBoost temperature regressor
│   ├── temperature_metrics.pkl     # Validation & test metrics
│   ├── rainfall_model.pkl          # Serialized XGBoost rainfall classifier
│   ├── rainfall_encoder.pkl        # Label encoder for rainfall classes
│   └── rainfall_metrics.pkl        # Validation & test metrics
│
├── data/
│   ├── raw/                        # Historical CSV datasets (50 Indian cities, 2010–2019)
│   └── processed/                  # Processed & feature-engineered datasets
│
├── requirements.txt                # Python package dependencies
├── launch_app.bat                  # One-click Windows launcher
├── install.bat                     # One-click dependency installer (Windows)
└── run_pipeline.bat                # Full ML pipeline runner (Windows)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10, 3.11, or 3.12
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/atharva-923/climate-ai.git
cd climate-ai
```

### 2. Create a Virtual Environment *(Recommended)*

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the App

**Option A — One-click launcher (Windows only):**
```cmd
launch_app.bat
```

**Option B — Command line (all platforms):**
```bash
streamlit run app/app.py
```

The app opens at **http://localhost:8501** in your browser.

> **Note:** Pre-trained models are included in `models/`. No retraining is required to run the app.

---

## ⚙️ Retraining the ML Pipeline

To rebuild the processed datasets and retrain models from scratch:

```bash
# Step 1: Clean and preprocess raw data
python src/preprocessing.py

# Step 2: Extract cyclical, rolling, and lag features
python src/features.py

# Step 3: Train the XGBoost Temperature Regressor
python src/train_temperature.py

# Step 4: Train the XGBoost Rainfall Classifier
python src/train_rainfall.py
```

> On Windows you can also run `run_pipeline.bat` to execute all steps in sequence.

---

## 📊 Model Performance

| Model | Task | Algorithm | Metric | Score |
|:------|:-----|:----------|:-------|:------|
| **Temperature** | Continuous Forecasting | XGBoost Regressor | MAE | **0.68°C – 0.86°C** |
| | | | R² Score | **92.7% – 97.5%** |
| | | | RMSE | **1.79°C** |
| **Rainfall** | 4-Tier Classification | XGBoost Classifier | Accuracy | **73.3%** |
| | | | Weighted F1 | **0.720** |

**Validation Scheme:** Out-of-time chronological split — Training on 2010–2022, Testing on unseen 2023–2025 data across all 50 cities.

---

## 🛠️ Technology Stack

| Layer | Technology |
|:------|:-----------|
| Application Framework | [Streamlit](https://streamlit.io) |
| Machine Learning | [XGBoost](https://xgboost.ai), [Scikit-learn](https://scikit-learn.org), [Joblib](https://joblib.readthedocs.io) |
| Live Weather & AQI | [Open-Meteo API](https://open-meteo.com) *(free, no API key required)* |
| Data Engineering | [Pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) |
| Visualizations | [Plotly](https://plotly.com), [Folium](https://python-visualization.github.io/folium), [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) |
| UI Design System | Obsidian Glassmorphism + Dynamic Aurora CSS Animation Engine |

---

## 🌍 Data Sources

| Dataset | Coverage | Records |
|:--------|:---------|:--------|
| Historical Daily Weather | 50 Indian cities, 2010–2019 | 182,500+ observations |
| Live Telemetry (Open-Meteo) | 50 Indian cities, real-time | Streamed on-demand |
| Air Quality (Open-Meteo AQI) | Major metros, real-time | Streamed on-demand |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'feat: add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

Please make sure your code follows the existing style and that new features include appropriate documentation.

---

## 📋 Roadmap

- [ ] Deploy to Streamlit Community Cloud
- [ ] Add lightning strike density overlay on radar map
- [ ] Extend coverage to additional South Asian cities
- [ ] Add cyclone / tropical storm track predictions
- [ ] REST API endpoint layer for programmatic access

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

## ⚠️ Disclaimer

This software is developed for educational, research, and demonstration purposes. **Do not use as a primary source for critical aviation dispatch, emergency navigation, or official weather forecasting and was Hackathin Topic.**

---

<div align="center">

Built with ❤️ for Indian meteorological intelligence and climate resilience.

**[⭐ Star this repo](https://github.com/atharva-923/climate-ai)** if you find it useful!

</div>
