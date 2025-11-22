# Malaria Early Warning System for Southern Mozambique

A climate-based malaria prediction dashboard built at Princeton's first "Vibe-a-thon" - demonstrating that technical experience is no longer a barrier to building impactful applications with AI.

## Overview

This application predicts malaria outbreak risk in Southern Mozambique (Gaza, Inhambane, and Maputo provinces) using climate variables like temperature, precipitation, and the Indian Ocean Subtropical Dipole (IOSD) index. Built entirely with Claude Sonnet 4.5 in under 6 hours.

### Key Features

- **Interactive Web Dashboard** - Streamlit-based interface for exploring data and predictions
- **Desktop GUI Application** - Tkinter-based tool for local analysis
- **ML-Powered Predictions** - Multiple regression models (Random Forest, Gradient Boosting, Huber) trained on 2020-2024 data
- **Geographic Visualization** - Interactive maps showing province-level malaria burden
- **Climate Correlation Analysis** - Explore relationships between climate variables and disease incidence

## Model Performance

| Province | Model | R² Score | MAPE | Correlation |
|----------|-------|----------|------|-------------|
| Gaza | Gradient Boosting | 0.999 | 3.6% | 0.999 |
| Inhambane | Random Forest | 0.124 | 21.7% | 0.804 |
| Maputo | Huber Regression | 0.653 | 21.9% | 0.818 |

*Note: Gaza 2024 showed a ~77% reduction in cases (likely due to intervention), which we modeled using a structural break approach.*

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone or download the project
cd Claude-Buildathon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Web Dashboard

```bash
# Local access only
streamlit run malaria_dashboard.py

# Network access (allow others on same WiFi to view)
streamlit run malaria_dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Open http://localhost:8501 in your browser.

### Running the Desktop GUI

```bash
python malaria_gui.py
```

### Sharing Publicly (via ngrok)

```bash
# Install ngrok if needed
brew install ngrok  # or download from ngrok.com

# Start tunnel
ngrok http 8501
```

Share the generated URL (e.g., `https://abc123.ngrok-free.dev`) with anyone!

## Project Structure

```
Claude-Buildathon/
├── malaria_dashboard.py     # Streamlit web dashboard
├── malaria_gui.py           # Desktop Tkinter GUI
├── retrain_models.py        # Model training script
├── requirements.txt         # Python dependencies
│
├── data/
│   ├── raw/                 # Original source files
│   └── processed/           # Cleaned data for models
│
├── models/
│   └── final_best_models.joblib
│
├── outputs/
│   ├── visualizations/      # Generated charts
│   └── exports/             # Exported CSV files
│
├── docs/                    # Documentation
└── archive/                 # Old iterations
```

## How It Works

### Data Pipeline
1. **Climate Data**: IOSD index, precipitation, temperature (with 1-3 month lags)
2. **Malaria Cases**: Monthly case counts by province (2020-2024)
3. **Feature Engineering**: Seasonal indicators, rolling averages, climate interactions

### Model Training
```bash
python retrain_models.py
```

This script:
- Loads processed climate-malaria data
- Engineers features (lags, interactions, seasonal encoding)
- Trains multiple model types per province
- Selects best performer based on test R²
- Saves models to `models/final_best_models.joblib`

### Key Climate Predictors
- **IOSD (Indian Ocean Subtropical Dipole)** - Strongest predictor of malaria seasonality
- **Precipitation** - 1-3 month lagged rainfall affects mosquito breeding
- **Temperature** - Influences mosquito lifecycle and parasite development

## Dashboard Views

1. **Overview Dashboard** - Key metrics and time series
2. **Geographic Map** - Interactive Folium map with province markers
3. **Time Series Analysis** - Historical trends with climate overlays
4. **Monthly Patterns** - Seasonal heatmaps and statistics
5. **Model Predictions** - 2024 actual vs predicted comparison
6. **Data Explorer** - Raw data access and export

## Built With

- **Claude Sonnet 4.5** - AI pair programmer for entire development
- **Streamlit** - Web dashboard framework
- **Tkinter** - Desktop GUI
- **scikit-learn** - Machine learning models
- **Plotly** - Interactive visualizations
- **Folium** - Geographic mapping
- **Pandas/NumPy** - Data processing

## About This Project

### Princeton Vibe-a-thon 2025

This project was built at Princeton's first "Vibe-a-thon" - a 6-hour buildathon proving that anyone with an idea can create functional applications using AI, regardless of technical background.

**Event Details:**
- Duration: 6 hours (10 AM - 4 PM)
- Building Time: ~4 hours
- Team Size: 1-5 people

**Mission:** Prove that technical barriers should not limit creativity. With AI tools like Claude, students of all backgrounds can build impactful solutions.

### The Challenge

Malaria remains a significant public health challenge in Mozambique. Early warning systems that leverage climate data can help health authorities prepare resources and interventions before outbreaks occur.

### What We Built

Starting with raw climate data and malaria case reports, we built:
- A complete ML pipeline for prediction
- Interactive dashboards for exploration
- Tools for public health decision support

All in a single afternoon, with Claude as our AI coding partner.

## Future Improvements

- [ ] Add forecast predictions for upcoming months
- [ ] Integrate real-time climate data APIs
- [ ] Mobile-responsive design
- [ ] Alert system for high-risk predictions
- [ ] Expand to additional provinces

## License

MIT License - Feel free to use, modify, and distribute.

## Acknowledgments

- **Goodbye Malaria** - For the climate and malaria data
- **Princeton SEAS & SML** - Faculty judges and support
- **Anthropic** - Claude AI and API credits
- **Princeton Vibe-a-thon Organizers** - Making this possible

---

*Built with Claude Sonnet 4.5 at Princeton's Vibe-a-thon 2025*
