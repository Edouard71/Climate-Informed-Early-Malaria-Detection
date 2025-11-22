# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Climate-linked Malaria Early Warning System for Southern Mozambique. Predicts monthly malaria incidence 1-2 months ahead using climate variables. Based on research by Philip Mwendwa (Goodbye Malaria, Summer 2025) analyzing climate-malaria relationships in Maputo Province.

**Goal**: Build an operational early warning system that leverages IOSD and precipitation to forecast malaria risk before peak transmission seasons.

## Key Research Findings

| Predictor | Effect Size | Best Lag | R² | Significance |
|-----------|-------------|----------|-----|--------------|
| **IOSD** | +0.4-0.5 cases/1000 per +1 SD | 0-2 months | 0.40-0.45 | p < 0.001 |
| **Precipitation** | +0.4 cases/1000 per +100mm | 1 month | 0.17-0.30 | p < 0.01 |
| **ENSO** | -0.09 to -0.18 per +1 ONI | weak | < 0.05 | not significant |

**Key insight**: IOSD (Indian Ocean) dominates over ENSO (Pacific) for this region. Positive IOSD → warm western Indian Ocean → enhanced rainfall → more mosquito breeding habitat.

## Development Commands

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the data pipeline
python build_master_dataset.py

# Run tests
pytest

# Type checking and linting
mypy src/
ruff check .
ruff format .
```

## Project Structure

```
├── malaria_model.py                     # Production model class (MalariaPredictor)
├── malaria_model.joblib                 # Trained model file
├── train_malaria_model.py               # Basic linear regression training
├── train_province_models.py             # Province-specific models
├── train_regime_model.py                # Regime-aware model with intervention detection
├── master_climate_malaria_provinces_2020_2024.csv  # Main dataset (180 rows)
├── predictions_vs_actuals_2024.png      # Model predictions visualization
├── model_performance_summary.png        # Performance metrics visualization
├── Goodbye Malaria Climate report Summer 2025.pdf  # Full research report
├── build_master_dataset.py              # ETL pipeline for data integration
├── malaria_early_warning_summary.md     # Analysis methodology
├── quick_reference_guide.md             # Operations guide
└── src/                                 # Application modules
```

## Master Dataset Schema (Long Format)

21 columns, one row per province per month:
- **Identifiers**: province, year, month, date
- **Health outcomes**: malaria_cases, malaria_incidence (per 1,000 pop)
- **Precipitation**: precip_mm, precip_lag1, precip_lag2, precip_lag3
- **Temperature**: temp_mean_celsius
- **IOSD**: iosd_index, iosd_lag1, iosd_lag2, iosd_lag3, iosd_phase
- **ENSO**: enso_oni, enso_lag1, enso_phase
- **Derived**: season, wet_season, population

## Data Sources

| Variable | Source | Resolution |
|----------|--------|------------|
| Malaria cases | DHIS2 / Goodbye Malaria | Monthly, province |
| Precipitation | CHIRPS v2.0 | ~5km, monthly |
| IOSD Index | ERA5 SST calculation | Regional, monthly |
| ENSO ONI | NOAA CPC | Global index, monthly |

## Key Constants

```python
PROVINCES = ['Maputo', 'Gaza', 'Inhambane']
POPULATION_ESTIMATES = {
    'Maputo': 2500000,
    'Gaza': 1450000,
    'Inhambane': 1560000
}
# Study period: Jan 2020 - Aug 2025
# Wet season: Nov-Apr (peak malaria Dec-Apr), Dry season: May-Oct (lowest Jun-Sep)
# IOSD thresholds: positive >+0.4°C, negative <-0.4°C
# ENSO thresholds: El Niño ≥+0.5°C, La Niña ≤-0.5°C

# Incidence calculation
# Incidence = (Cases / Population) × 1000
```

## Climate Index Definitions

**IOSD (Indian Ocean Subtropical Dipole)**: Calculated from ERA5 SSTs as difference between western (20°S–10°S, 50–70°E) and eastern (20°S–10°S, 90–110°E) box averages. Positive phase → enhanced rainfall over southern Mozambique.

**ENSO ONI**: NOAA Oceanic Niño Index (3-month Niño 3.4 SST anomaly). El Niño tends to suppress rainfall in this region, but signal is weak compared to IOSD.

## Production Model: Regime-Aware Predictor

The final model (`malaria_model.py`) uses a regime-aware approach that handles structural breaks (interventions):

```python
from malaria_model import MalariaPredictor

# Load trained model
model = MalariaPredictor.load('malaria_model.joblib')

# Make predictions
predictions = model.predict(new_data)
```

### Model Features

| Feature | Type | Description |
|---------|------|-------------|
| `iosd_lag1` | Climate | IOSD index from 1 month ago (strongest predictor) |
| `iosd_lag2` | Climate | IOSD index from 2 months ago |
| `precip_lag1` | Climate | Precipitation from 1 month ago |
| `baseline_ratio_lag1` | Regime | Last month's cases / historical mean |
| `recent_low_regime` | Regime | Binary: 1 if recent 3-month avg < 50% of baseline |
| `Season` | Categorical | Wet or Dry |
| `Province` | Categorical | Gaza, Inhambane, or Maputo |

### Model Performance (Test Year: 2024)

| Metric | Train | Test |
|--------|-------|------|
| R² | 0.88 | 0.55-0.63 |
| RMSE | 9,516 | 14,132 |
| MAPE | 42% | 151% |

**Note**: Gaza showed an 80% case reduction in 2024 (likely intervention), causing high test error. Maputo performs best with ~27% MAPE.

### Regime Detection

The model automatically detects intervention periods:
- Historical baselines: Gaza ~32K, Inhambane ~66K, Maputo ~4K monthly cases
- "Low regime" flagged when cases drop >50% below historical baseline
- Regime features allow model to adapt to intervention effects

## Alert Thresholds

| Risk Level | Incidence (per 1,000) |
|------------|----------------------|
| Low | < 2.0 |
| Moderate | 2.0 - 3.5 |
| High | 3.5 - 5.0 |
| Very High | > 5.0 |

## Seasonal Pattern

- **Peak transmission**: December - April (wet season)
- **Lowest incidence**: June - September (dry season)
- **Malaria peaks 0-2 months after rainfall peaks**

## Known Limitations

- 5-year time series limits statistical power, especially for ENSO
- Gaza 2024 shows 80% case reduction - likely intervention effect not captured by climate
- Regime features help but require 1-3 months of data to detect structural breaks
- Model assumes historical baselines remain valid; major demographic changes would require recalibration

## Model Evolution

| Version | Approach | Test R² | Notes |
|---------|----------|---------|-------|
| v1 | Pooled linear regression | 0.32 | Baseline |
| v2 | Province-specific models | 0.30 | No improvement |
| v3 | Season interactions | 0.27 | Worse |
| v4 | **Regime-aware** | **0.63** | Best - handles interventions |
