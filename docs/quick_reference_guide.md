# Climate-Malaria Early Warning System: Quick Reference Guide
## Southern Mozambique | Maputo, Gaza, Inhambane Provinces

---

## 🎯 Executive Summary

**Goal**: Predict malaria risk 1-2 months ahead using climate data

**Best Predictors**:
1. **IOSD** (Indian Ocean Subtropical Dipole) → R² = 0.45
2. **Precipitation** (CHIRPS) → R² = 0.30
3. ~~ENSO~~ (too weak for this region) → R² < 0.07

**Key Insight**: Focus operational forecasts on IOSD + precipitation at 1-2 month lags.

---

## 📊 Variables Summary

### Variables Already Used

| Variable | Source | Resolution | Effect Size | Best Lag |
|----------|--------|------------|-------------|----------|
| **Malaria Incidence** | DHIS2/Goodbye Malaria | Monthly, province | Target variable | - |
| **Precipitation** | CHIRPS v2.0 | ~5km, monthly | +0.3-0.5 cases/1,000 per +100mm | 1 month |
| **IOSD Index** | ERA5 SST | Regional, monthly | +0.48 cases/1,000 per +1 SD | 1 month |
| **ENSO (ONI)** | NOAA CPC | Global, 3-month avg | -0.14 cases/1,000 per +1 unit | Weak/inconsistent |

### Variables to Add (Recommended)

| Variable | Priority | Source | Purpose |
|----------|----------|--------|---------|
| Temperature (mean) | HIGH | ERA5-Land | Direct mosquito/parasite development |
| ITN coverage | HIGH | Program data | Control for interventions |
| IRS timing | MEDIUM | Program data | Control for interventions |
| Humidity/dew point | MEDIUM | ERA5-Land | Mosquito survival |
| Soil moisture | LOW | ERA5-Land | Additional breeding habitat proxy |

---

## 🔑 Important Short-Term Predictors (0-2 Month Lags)

### 1. IOSD (⭐⭐⭐ STRONGEST)

**Effect**: +0.4 to +0.5 cases per 1,000 per +1 SD increase
**Optimal lag**: 1 month
**Statistical significance**: p < 0.001
**Model fit**: R² = 0.40-0.45

**How it works**:
- Positive IOSD → warmer western Indian Ocean → enhanced rainfall over Mozambique
- Negative IOSD → cooler waters → reduced rainfall

**Classification**:
- Positive: IOSD > +0.4°C (high malaria risk)
- Neutral: -0.4°C to +0.4°C (moderate risk)
- Negative: IOSD < -0.4°C (low risk)

**Operational use**: Monitor IOSD monthly; issue alerts when positive phase during Oct-Jan.

---

### 2. Precipitation (⭐⭐ STRONG)

**Effect**: +0.3 to +0.5 cases per 1,000 per +100mm
**Optimal lag**: 1 month
**Statistical significance**: p < 0.01
**Model fit**: R² = 0.17-0.30

**How it works**:
- Heavy rainfall → standing water → mosquito breeding → increased transmission

**Temporal pattern**:
- Strong relationship: 2020-2022 (r > 0.80)
- Weaker relationship: 2023-2024 (r ≈ 0.30-0.60) ← investigate why

**Operational use**: Track monthly rainfall totals; alert when >200mm in wet season months.

---

### 3. ENSO (⭐ WEAK)

**Effect**: -0.09 to -0.18 cases per 1,000 per +1 ONI unit
**Optimal lag**: None clear
**Statistical significance**: p > 0.05 (not significant)
**Model fit**: R² < 0.05

**Why weak**: Indian Ocean (IOSD) dominates climate variability in this region more than Pacific (ENSO).

**Operational use**: Consider as secondary indicator only; do not rely on ENSO alone.

---

## 📋 Master Dataset Schema

### Format: Long (One Row Per Province Per Month)

```
province | year | month | date       | malaria_cases | malaria_incidence | ...
---------|------|-------|------------|---------------|-------------------|----
Maputo   | 2020 | 1     | 2020-01-01 | 7638          | 3.055             | ...
Maputo   | 2020 | 2     | 2020-02-01 | 7318          | 2.927             | ...
Gaza     | 2020 | 1     | 2020-01-01 | [value]       | [value]           | ...
```

### Core Columns (21 total)

**Identifiers (4)**:
- `province`: String [Maputo, Gaza, Inhambane]
- `year`: Integer [2020-2025]
- `month`: Integer [1-12]
- `date`: Date [YYYY-MM-01]

**Health Outcomes (2)**:
- `malaria_cases`: Integer (raw count)
- `malaria_incidence`: Float (per 1,000 population)

**Precipitation (4)**:
- `precip_mm`: Float (monthly total)
- `precip_lag1`: Float (1 month prior)
- `precip_lag2`: Float (2 months prior)
- `precip_lag3`: Float (3 months prior)

**IOSD (5)**:
- `iosd_index`: Float (°C)
- `iosd_lag1`: Float
- `iosd_lag2`: Float
- `iosd_lag3`: Float
- `iosd_phase`: Categorical [positive, negative, neutral]

**ENSO (3)**:
- `enso_oni`: Float (dimensionless)
- `enso_lag1`: Float
- `enso_phase`: Categorical [el_nino, la_nina, neutral]

**Derived (3)**:
- `season`: Categorical [wet, dry]
- `wet_season`: Binary [1, 0]
- `population`: Integer

---

## 🛠️ Implementation Checklist

### Phase 1: Data Assembly (Weeks 1-2)
- [ ] Gather malaria case data for Gaza and Inhambane
- [ ] Confirm population estimates for each province
- [ ] Download CHIRPS precipitation for all provinces (2020-2025)
- [ ] Calculate/download IOSD index time series
- [ ] Download ENSO ONI from NOAA

### Phase 2: Master Dataset Creation (Weeks 3-4)
- [ ] Convert malaria cases to incidence rates
- [ ] Extract province-mean precipitation from CHIRPS
- [ ] Merge all data sources by province and date
- [ ] Create lagged variables (1-3 months)
- [ ] Add seasonal indicators
- [ ] Validate data completeness and continuity

### Phase 3: Model Development (Weeks 5-8)
- [ ] Train baseline model: `malaria ~ iosd_lag1 + precip_lag1 + season`
- [ ] Test province-specific vs. mixed-effects models
- [ ] Validate on 2024-2025 data
- [ ] Calculate prediction accuracy metrics (RMSE, MAE, R²)
- [ ] Define alert thresholds (e.g., predicted incidence >3.5 = "high risk")

### Phase 4: Operational System (Months 3-6)
- [ ] Automate monthly data updates
- [ ] Create prediction dashboard
- [ ] Develop standard monthly forecast report template
- [ ] Train staff on interpreting forecasts
- [ ] Pilot test with program teams

---

## 🚀 Recommended Prediction Model (Baseline)

```R
# Simple operational model for immediate use
library(lme4)

model <- lmer(
  malaria_incidence ~ 
    iosd_lag1 +           # Strongest predictor (+0.48 cases/1,000 per SD)
    iosd_lag2 +           # Still significant (+0.30 cases/1,000 per SD)
    precip_lag1 +         # Direct rainfall effect (+0.49 cases/1,000 per 100mm)
    season +              # Wet vs dry season baseline
    (1 | province),       # Random intercept for each province
  data = master_df
)

# Make predictions for next month
predict(model, newdata = next_month_climate_data)
```

### Alert Thresholds (Example - to be refined)

| Predicted Incidence | Risk Level | Action |
|---------------------|------------|--------|
| < 2.0 | **Low** | Routine surveillance |
| 2.0 - 3.5 | **Moderate** | Enhanced monitoring |
| 3.5 - 5.0 | **High** | Pre-position resources |
| > 5.0 | **Very High** | Activate response protocols |

---

## 📈 Expected Model Performance

Based on your analysis:

- **Best case** (IOSD at lag 1): R² = 0.45, explains 45% of variance
- **Multi-variable model** (IOSD + precip + season): Expected R² = 0.50-0.60
- **Practical accuracy**: Should predict within ±1.0 cases/1,000 for 70% of months

### Known Limitations
- Short time series (5 years) limits long-term validation
- Does not account for interventions (ITN, IRS)
- Province-level resolution may miss district heterogeneity
- 2023-2024 show weaker precipitation relationships (investigate)

---

## 🗓️ Typical Seasonal Pattern

```
Month    | Season | Avg Precip | Avg Incidence | Risk Level
---------|--------|------------|---------------|------------
January  | Wet    | High       | High          | High
February | Wet    | High       | Peak          | Very High
March    | Wet    | High       | High          | High
April    | Wet    | Medium     | Medium        | Moderate
May      | Dry    | Low        | Low-Med       | Moderate
June     | Dry    | Very Low   | Low           | Low
July     | Dry    | Very Low   | Lowest        | Low
August   | Dry    | Very Low   | Low           | Low
September| Dry    | Low        | Low           | Low
October  | Dry    | Low        | Medium        | Moderate
November | Wet    | Medium     | Medium        | Moderate
December | Wet    | High       | High          | High
```

**Key insight**: Malaria peaks 0-2 months after rainfall peaks (Jan-Feb rainfall → Feb-Mar malaria peak)

---

## 🔍 Quality Checks for Master Dataset

Before modeling, verify:

1. **Completeness**:
   - All provinces have data for all months in range?
   - No unexpected gaps in time series?

2. **Date alignment**:
   - All dates standardized to 1st of month?
   - No duplicate province-date combinations?

3. **Lag calculations**:
   - Does `precip_lag1` match previous month's `precip_mm`?
   - Do lags respect province boundaries (no cross-province leakage)?

4. **Missing data**:
   - How many missing values per column?
   - Are missings random or systematic?
   - Never forward-fill climate indices (loses predictive value)

5. **Outliers**:
   - Any extreme values (>99th percentile) to investigate?
   - Cyclone Freddy (Feb 2023) will show as outlier in precipitation

---

## 📞 Key Data Sources & Access

### Malaria Data
- **Source**: Goodbye Malaria / MISAU DHIS2
- **Format**: Excel/CSV with monthly cases by province
- **Update frequency**: Monthly (2-4 week delay)
- **Contact**: [M&E team]

### Precipitation (CHIRPS)
- **Website**: https://data.chc.ucsb.edu/products/CHIRPS-2.0/
- **Format**: GeoTIFF rasters
- **Resolution**: 0.05° (~5km)
- **Update frequency**: Monthly (1-2 week delay)
- **Access**: Public, direct download

### IOSD Index
- **Source**: Calculated from ERA5 SST
- **Calculation**: SST_west (50-70°E) - SST_east (90-110°E), 20-10°S
- **Alternative**: Pre-computed indices from climate labs
- **Update frequency**: Monthly

### ENSO ONI
- **Website**: https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
- **Format**: Text table
- **Update frequency**: Monthly
- **Access**: Public, NOAA Climate Prediction Center

---

## 💡 Tips for Success

1. **Start simple**: IOSD + precipitation model first, add complexity later
2. **Validate continuously**: Re-check predictions against actual cases monthly
3. **Update annually**: Re-train model as new data accumulates
4. **Communicate clearly**: Translate probabilities into actionable risk levels
5. **Document everything**: Keep detailed notes on data processing decisions
6. **Collaborate**: Share findings with climate scientists for validation

---

## 🎓 Next Learning Steps

### To deepen understanding:
- Read Ikeda et al. (2017) on seasonally lagged climate-malaria models
- Explore ECMWF seasonal forecasts for IOSD/precipitation
- Learn about mixed-effects models for multi-province data
- Study epidemiological time series methods (ARIMA, state-space models)

### To improve system:
- Add temperature to models (mosquito development rate)
- Incorporate intervention timing (ITN campaigns, IRS)
- Test machine learning approaches (random forests, XGBoost)
- Develop district-level resolution models
- Integrate real-time climate forecasts (IRI, ECMWF)

---

## 📚 Key Reference

**Your Study**: "Linking Climate Variability and Malaria Trends: A Case Study on Southern Mozambique"  
**Author**: Philip Mwendwa (M&E Intern, Summer 2025)  
**Main Finding**: IOSD is strongest predictor (R²=0.45), precipitation is second (R²=0.30), ENSO is weak (R²<0.07)

---

## ⚠️ Important Reminders

- Climate models **support** decisions, they don't **replace** programmatic expertise
- Predictions are probabilistic; always maintain flexibility in response
- Intervention data (ITN, IRS) should be added when available
- Model performance should be monitored and updated regularly
- This system complements (not replaces) traditional surveillance

---

*Last Updated: November 2025*  
*For questions or suggestions, contact: [Your contact info]*
