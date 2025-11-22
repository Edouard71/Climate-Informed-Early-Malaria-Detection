# Climate-Informed Malaria Early Warning System: Analysis Summary
## Southern Mozambique (Maputo, Gaza, Inhambane Provinces)

---

## 1. Variables Already Used in Your Analysis

### Health Outcome Variable
- **Malaria Incidence**: Cases per 1,000 population (monthly)
  - Raw case counts converted using assumed population of 2.5M for Maputo Province
  - Study period: January 2020 - August 2025
  - Aggregated from 7 districts to province-level monthly series

### Climate Variables

#### Direct Climate Variables (Primary)
1. **Precipitation** (CHIRPS dataset)
   - Resolution: ~0.05° (~5km)
   - Monthly totals (mm)
   - Cropped to province boundary, masked to land
   - Aggregated to provincial mean
   - Effects scaled per +100mm

2. **Historical Precipitation Context** (WorldClim v2.1)
   - Long-term averages (1970-2000)
   - Used for seasonal climatology baseline

#### Climate Variables Explored (Secondary)
3. **Temperature** (ERA5-Land)
4. **Dew Point** (ERA5-Land)
5. **Wind** (ERA5-Land)
6. **Soil Moisture** (ERA5-Land)
7. **Solar Radiation** (ERA5-Land)

### Large-Scale Climate Indices

#### Indian Ocean Subtropical Dipole (IOSD)
- **Source**: Calculated from ERA5 sea surface temperatures (SSTs)
- **Calculation**: SST difference between western box (20°S-10°S, 50-70°E) and eastern box (20°S-10°S, 90-110°E)
- **Phase Classification**: ±0.4°C threshold
  - Positive phase: Enhanced summer rainfall
  - Negative phase: Reduced rainfall
  - Neutral: Between thresholds
- **Effects scaled**: Per +1 standard deviation (SD ≈ 0.9°C)
- **Standardization**: Z-score normalization used for correlation analysis

#### El Niño-Southern Oscillation (ENSO)
- **Source**: NOAA Oceanic Niño Index (ONI) v5
- **Measure**: 3-month running mean of Niño 3.4 SST anomalies
- **Phase Classification**: 
  - El Niño: +0.5°C or higher (typically drier in Southern Mozambique)
  - La Niña: -0.5°C or lower (typically wetter)
  - Neutral: Between -0.5°C and +0.5°C
- **Effects scaled**: Per +1 ONI unit

### Analytical Variables
- **Time lags**: 0-6 months tested for all climate predictors
- **Temporal resolution**: Monthly (all variables time-aligned to 1st of month)

---

## 2. Important Short-Term Predictors (0-2 Month Lags)

### **Top Priority: IOSD (Indian Ocean Subtropical Dipole)**
- **Effect size**: +0.4 to +0.5 cases per 1,000 population per +1 SD increase
- **Strongest lags**: 0-2 months (r ≈ 0.4-0.55)
- **Statistical significance**: p < 0.001 at lags 0-2
- **Model fit**: R² = 0.40 at lag 0, R² = 0.45 at lag 1
- **Practical importance**: Clear, immediate signal that reverses after 4 months
- **Interpretation**: Positive IOSD → enhanced rainfall → increased breeding habitat → higher transmission

**Key Finding**: IOSD is the strongest standalone predictor identified in your study.

---

### **High Priority: Precipitation**
- **Effect size**: +0.3 to +0.5 cases per 1,000 population per +100mm
- **Strongest correlations**: 1-month lag (r = 0.32-0.89 across years)
- **Consistency**: Strong positive relationship in 2020-2022, weaker in 2023-2024
- **Statistical significance**: p < 0.01 at lags 0-2
- **Model fit**: R² = 0.17-0.30 at lags 0-2
- **Note**: Relationship strength varies by year:
  - 2020-2021: Very strong (r > 0.80)
  - 2022: Strong (r ≈ 0.88 at 1-month lag)
  - 2023-2024: Moderate to weak (r ≈ 0.30-0.60)

**Key Finding**: Most immediate climate variable; best at 1-month lag.

---

### **Lower Priority: ENSO**
- **Effect size**: -0.09 to -0.18 cases per 1,000 per +1 ONI unit
- **Correlations**: Weak negative (r ≈ -0.10 to -0.25)
- **Statistical significance**: p > 0.05 at most lags
- **Model fit**: R² < 0.05 at all lags
- **Interpretation**: El Niño conditions (positive ONI) → slightly reduced rainfall → marginally lower transmission
- **Practical importance**: LIMITED in this study period/region

**Key Finding**: Weak predictor for Maputo; IOSD dominates regional climate variability.

---

### Temporal Patterns Observed
- **Peak malaria season**: December-April (wet season)
- **Lowest incidence**: June-October (dry season)
- **Lag structure**: Climate effects are rapid
  - 0-2 months: Strong positive effects
  - 3+ months: Effects decline, often reverse
- **Seasonal dependence**: Strongest climate-malaria relationships during wet season onset

---

## 3. Master Dataset Schema for Multi-Province Modeling

### Recommended Structure: Long Format (One Row Per Province Per Month)

```
+-------------+------+------+------------+------------------+-----------+----------+-----------+
| province    | year | month| date       | malaria_cases    | malaria   | precip   | precip    |
|             |      |      |            |                  | _incidence| _mm      | _lag1     |
+-------------+------+------+------------+------------------+-----------+----------+-----------+
| Maputo      | 2020 | 1    | 2020-01-01 | 7638             | 3.055     | 245.3    | 189.2     |
| Maputo      | 2020 | 2    | 2020-02-01 | 7318             | 2.927     | 198.7    | 245.3     |
| Gaza        | 2020 | 1    | 2020-01-01 | [cases]          | [rate]    | 267.8    | 201.5     |
| Gaza        | 2020 | 2    | 2020-02-01 | [cases]          | [rate]    | 223.4    | 267.8     |
| Inhambane   | 2020 | 1    | 2020-01-01 | [cases]          | [rate]    | 198.5    | 175.3     |
...
+-------------+------+------+------------+------------------+-----------+----------+-----------+

+----------+-----------+-----------+--------+----------+----------+---------+---------+
| precip   | temp_mean | temp_min  | temp   | iosd     | iosd_lag1| enso_oni| enso    |
| _lag2    | _celsius  | _celsius  | _max   | _index   |          |         | _lag1   |
+----------+-----------+-----------+--------+----------+----------+---------+---------+
| 156.8    | 26.4      | 21.2      | 31.5   | 0.82     | 0.65     | 0.5     | 0.3     |
| 189.2    | 25.8      | 20.8      | 30.9   | 0.95     | 0.82     | 0.6     | 0.5     |
| 148.2    | 27.1      | 22.0      | 32.1   | 0.88     | 0.71     | 0.5     | 0.3     |
| 201.5    | 26.5      | 21.5      | 31.4   | 1.02     | 0.88     | 0.6     | 0.5     |
| 162.3    | 26.8      | 21.8      | 31.8   | 0.79     | 0.68     | 0.5     | 0.3     |
...
+----------+-----------+-----------+--------+----------+----------+---------+---------+

+--------+----------+-----------+------------+-------------+
| season | wet      | population| intervention| intervention|
|        | _season  |           | _itns       | _irs        |
+--------+----------+-----------+------------+-------------+
| wet    | 1        | 2500000   | NA         | NA          |
| wet    | 1        | 2500000   | NA         | NA          |
| wet    | 1        | 1450000   | NA         | NA          |
| wet    | 1        | 1450000   | NA         | NA          |
| wet    | 1        | 1560000   | NA         | NA          |
...
+--------+----------+-----------+------------+-------------+
```

---

### Core Variable Definitions

#### Geographic and Temporal Identifiers
- **province**: String [Maputo, Gaza, Inhambane]
- **year**: Integer [2020-2025]
- **month**: Integer [1-12]
- **date**: Date [YYYY-MM-01] (standardized to 1st of month)

#### Health Outcomes
- **malaria_cases**: Integer (raw monthly case count)
- **malaria_incidence**: Float (cases per 1,000 population)

#### Precipitation Variables
- **precip_mm**: Float (monthly total precipitation in mm, province mean)
- **precip_lag1**: Float (precipitation from 1 month prior)
- **precip_lag2**: Float (precipitation from 2 months prior)
- **precip_lag3**: Float (optional: 3 months prior)
- **precip_anomaly**: Float (optional: deviation from long-term monthly mean)

#### Temperature Variables (Optional but Recommended)
- **temp_mean_celsius**: Float (monthly mean temperature)
- **temp_min_celsius**: Float (monthly minimum temperature)
- **temp_max_celsius**: Float (monthly maximum temperature)
- **temp_lag1**: Float (mean temperature from 1 month prior)

#### Climate Indices
- **iosd_index**: Float (raw IOSD index value, °C)
- **iosd_lag1**: Float (IOSD from 1 month prior)
- **iosd_lag2**: Float (IOSD from 2 months prior)
- **iosd_phase**: Categorical [positive, negative, neutral] (based on ±0.4°C threshold)
- **enso_oni**: Float (Oceanic Niño Index)
- **enso_lag1**: Float (ONI from 1 month prior)
- **enso_lag2**: Float (ONI from 2 months prior)
- **enso_phase**: Categorical [el_nino, la_nina, neutral] (based on ±0.5°C threshold)

#### Derived Variables
- **season**: Categorical [wet, dry]
  - Wet: November-April
  - Dry: May-October
- **wet_season**: Binary [1 = wet season, 0 = dry season]
- **month_name**: String [January, February, ...] (optional for readability)

#### Contextual Variables
- **population**: Integer (province population estimate)
- **intervention_itns**: Float (optional: insecticide-treated net coverage %, if available)
- **intervention_irs**: Float (optional: indoor residual spraying coverage %, if available)
- **intervention_notes**: String (optional: text field for major interventions/events)

---

### Key Design Principles

#### 1. **Long Format (Tidy Data)**
- One row per province per month
- Facilitates mixed-effects models with province as random effect
- Easy to filter, aggregate, and visualize
- Compatible with most statistical software (R, Python, Stata)

#### 2. **Explicit Lag Variables**
- Pre-compute lagged predictors (lag1, lag2) rather than calculating on-the-fly
- Improves code clarity and reproducibility
- Easier to validate temporal alignment
- Supports rapid model iteration

#### 3. **Standardized Date Format**
- All dates aligned to 1st of month (YYYY-MM-01)
- Consistent with your current methodology
- Simplifies merging across datasets
- Avoids end-of-month misalignment issues

#### 4. **Province-Level Spatial Resolution**
- Climate variables aggregated to province mean
- Matches administrative health reporting units
- Computationally efficient for early warning systems
- Can be refined to district-level if needed later

#### 5. **Missingness Strategy**
- Use NA for truly missing values
- Document reasons for missing data
- Consider imputation only for short gaps (<2 months)
- Never forward-fill climate indices (loses predictive value)

---

### Alternative Schema Options

#### Option A: Wide Format (One Row Per Month, Provinces as Columns)
```
date       | maputo_cases | gaza_cases | inhambane_cases | maputo_incidence | ...
2020-01-01 | 7638         | [value]    | [value]         | 3.055            | ...
```
**Pros**: Compact, easy to view all provinces at once
**Cons**: Difficult to model with province-level random effects; harder to add provinces

#### Option B: Separate Files Per Province
```
maputo_monthly.csv
gaza_monthly.csv
inhambane_monthly.csv
```
**Pros**: Simpler data structure, smaller files
**Cons**: Complicates multi-province analysis; harder to maintain consistency

**Recommendation**: Use long format (main schema) for modeling; wide format can be created for visualization as needed.

---

## 4. Data Integration Workflow

### Step 1: Consolidate Malaria Data
```python
# Pseudocode
malaria_data = []
for province in ['Maputo', 'Gaza', 'Inhambane']:
    # Load case data
    df = load_province_cases(province)
    # Add province identifier
    df['province'] = province
    # Standardize date to first of month
    df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
    # Calculate incidence
    df['malaria_incidence'] = (df['cases'] / population[province]) * 1000
    malaria_data.append(df)

malaria_df = pd.concat(malaria_data)
```

### Step 2: Process Climate Variables
```python
# For each province boundary:
for province in provinces:
    # Crop CHIRPS raster to province
    precip = crop_and_mask(chirps_raster, province_boundary)
    # Calculate spatial mean
    precip_mean = precip.mean()
    # Store with date and province
    climate_data.append({'province': province, 'date': date, 'precip_mm': precip_mean})
```

### Step 3: Add Climate Indices
```python
# IOSD and ENSO are the same for all provinces (regional indices)
climate_indices = load_iosd_enso_data()
# Replicate for each province
for province in provinces:
    climate_indices_province = climate_indices.copy()
    climate_indices_province['province'] = province
    all_indices.append(climate_indices_province)
```

### Step 4: Create Lagged Variables
```python
# Create lags by province
master_df = master_df.sort_values(['province', 'date'])
for lag in [1, 2, 3]:
    master_df[f'precip_lag{lag}'] = master_df.groupby('province')['precip_mm'].shift(lag)
    master_df[f'iosd_lag{lag}'] = master_df.groupby('province')['iosd_index'].shift(lag)
    master_df[f'enso_lag{lag}'] = master_df.groupby('province')['enso_oni'].shift(lag)
```

### Step 5: Merge Everything
```python
master_df = malaria_df.merge(climate_df, on=['province', 'date'], how='left')
master_df = master_df.merge(indices_df, on=['province', 'date'], how='left')
# Add derived variables
master_df['season'] = master_df['month'].apply(classify_season)
master_df['wet_season'] = (master_df['season'] == 'wet').astype(int)
```

---

## 5. Modeling Recommendations

### Baseline Model (Province-Specific)
```R
# Separate model per province
lm(malaria_incidence ~ precip_lag1 + iosd_lag1 + season, data = maputo_data)
```

### Mixed-Effects Model (All Provinces)
```R
# Random intercept for province
lmer(malaria_incidence ~ precip_lag1 + iosd_lag1 + season + (1|province), data = master_df)
```

### Operational Early Warning Model
```R
# Focus on short-term predictors (0-2 month lags)
model <- lm(malaria_incidence ~ 
              iosd_lag1 + iosd_lag2 +           # Strongest predictor
              precip_lag1 + precip_lag2 +       # Direct rainfall effect
              season +                           # Seasonal baseline
              province,                          # Province fixed effects
            data = master_df)
```

### Model Validation Strategy
- **Training period**: 2020-2023
- **Testing period**: 2024-2025
- **Cross-validation**: Leave-one-year-out
- **Metrics**: RMSE, MAE, correlation between predicted and observed

---

## 6. Priority Actions for System Development

### Immediate (Next 1-2 Months)
1. ✅ Build master dataset for Maputo Province (you've already started)
2. ⏳ Extend to Gaza and Inhambane provinces
3. ⏳ Implement lagged variable creation pipeline
4. ⏳ Develop baseline prediction model with IOSD + precipitation

### Short-Term (3-6 Months)
5. Add temperature variables (mean, min, max)
6. Test mixed-effects models across provinces
7. Validate model predictions for 2024-2025
8. Create automated data update pipeline
9. Develop simple visualization dashboard

### Medium-Term (6-12 Months)
10. Incorporate intervention data (ITN coverage, IRS campaigns)
11. Refine district-level predictions (if data available)
12. Test ensemble modeling approaches
13. Develop operational alert thresholds
14. Create monthly forecasting reports

### Long-Term (12+ Months)
15. Integrate real-time climate forecasts (e.g., IRI, ECMWF)
16. Expand to other provinces in Mozambique
17. Develop mobile-friendly early warning interface
18. Publish methodology and validation results

---

## 7. Data Quality Considerations

### Known Limitations from Your Study
- **Short time series**: Only 5 years (2020-2024) limits statistical power
- **ENSO variability**: Low ENSO variability during study period may underestimate its importance
- **Population estimates**: Static population assumption (2.5M for Maputo)
- **Intervention data**: Not included in current models
- **District-level heterogeneity**: Aggregated to province level

### Recommendations for Improvement
- **Extend time series**: Acquire historical data back to 2010-2015 if possible
- **Update population**: Use annual population projections by province
- **Add interventions**: Incorporate ITN distribution, IRS timing, case management changes
- **District resolution**: Model at district level for higher spatial precision
- **Data validation**: Cross-check malaria case data with HMIS/DHIS2 for consistency

---

## 8. Key Takeaways for Early Warning System

### What Works (Based on Your Analysis)
✅ **IOSD at 1-2 month lag** → Most reliable short-term predictor
✅ **Precipitation at 1 month lag** → Strongest direct climate variable
✅ **Seasonal patterns** → Consistent December-April peak
✅ **Province-level aggregation** → Appropriate for operational use

### What Doesn't Work Well
❌ **ENSO alone** → Too weak for Maputo region
❌ **Long lags (>3 months)** → Effects reverse or become unreliable
❌ **2024 precipitation** → Weaker relationship than earlier years (investigate why)

### Critical Success Factors
1. **Real-time IOSD monitoring** → Must track southwestern Indian Ocean SSTs monthly
2. **Precipitation nowcasting** → CHIRPS data available with ~1 week delay (operational)
3. **Seasonal context** → Always interpret predictions relative to wet/dry season
4. **Model updating** → Re-train annually as new data accumulates
5. **Communication** → Translate statistical predictions into actionable alerts for health teams

---

## Contact & Next Steps

**Author**: Philip Mwendwa (M&E Intern, May-August 2025)
**Study Region**: Southern Mozambique (Maputo, Gaza, Inhambane)
**Current Status**: Proof-of-concept completed for Maputo Province

### Recommended Next Conversations
- Discuss Gaza and Inhambane data availability
- Review population estimates for incidence calculations
- Align on alert threshold definitions (e.g., "high risk" = incidence > X)
- Plan quarterly model validation schedule

---

*This summary synthesizes findings from "Linking Climate Variability and Malaria Trends: A Case Study on Southern Mozambique" (Summer 2025)*
