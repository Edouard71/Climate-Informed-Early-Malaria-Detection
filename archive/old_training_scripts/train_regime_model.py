"""
Malaria Early Warning System - Regime Change Model
===================================================
Handles structural breaks (interventions, reporting changes) by:
1. Detecting anomalous periods using historical baselines
2. Adding regime indicators to capture intervention effects
3. Training a model that can adapt to different transmission regimes

Key insight: Gaza dropped 80% in 2024, Inhambane dropped 34%.
These aren't climate effects - they're likely interventions.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare data."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    lag_cols = ['precip_lag1', 'precip_lag2', 'iosd_lag1', 'iosd_lag2']
    df = df.dropna(subset=lag_cols)
    return df


def detect_regime_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect structural breaks by comparing to historical baseline.
    A regime change is flagged when cases drop significantly from
    the province's historical average.
    """
    df = df.copy()

    # Calculate historical baseline (2020-2023) per province
    historical = df[df['Year'] < 2024].groupby('Province')['malaria_cases'].agg(['mean', 'std'])
    historical.columns = ['hist_mean', 'hist_std']

    print("="*60)
    print("REGIME CHANGE DETECTION")
    print("="*60)
    print("\nHistorical Baselines (2020-2023):")
    print(historical.round(0))

    # Merge historical stats
    df = df.merge(historical, left_on='Province', right_index=True)

    # Calculate z-score: how many SDs below historical mean?
    df['zscore'] = (df['malaria_cases'] - df['hist_mean']) / df['hist_std']

    # Regime indicators
    # Low regime: cases are more than 1.5 SD below historical mean
    df['regime_low'] = (df['zscore'] < -1.5).astype(int)

    # Calculate regime by province-year
    regime_summary = df.groupby(['Province', 'Year']).agg({
        'malaria_cases': 'mean',
        'hist_mean': 'first',
        'regime_low': 'mean'  # Proportion of months in low regime
    }).round(2)
    regime_summary['pct_of_baseline'] = (regime_summary['malaria_cases'] / regime_summary['hist_mean'] * 100).round(0)

    print("\nRegime Detection by Province-Year:")
    print(regime_summary[['malaria_cases', 'pct_of_baseline', 'regime_low']])

    # Flag province-years with >50% of months in low regime
    print("\n\nDetected Regime Changes (>50% months below threshold):")
    anomalies = regime_summary[regime_summary['regime_low'] > 0.5]
    if len(anomalies) > 0:
        for idx in anomalies.index:
            prov, year = idx
            pct = anomalies.loc[idx, 'pct_of_baseline']
            print(f"  - {prov} {year}: {pct:.0f}% of baseline (INTERVENTION LIKELY)")
    else:
        print("  None detected")

    return df


def create_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that capture regime changes.
    """
    df = df.copy()

    # Rolling average of recent cases (captures regime shifts)
    # For each province, calculate 3-month rolling mean
    df = df.sort_values(['Province', 'date'])
    df['cases_rolling_3m'] = df.groupby('Province')['malaria_cases'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # Ratio to historical baseline
    df['baseline_ratio'] = df['malaria_cases'] / df['hist_mean']

    # Lagged baseline ratio (what was the regime last month?)
    df['baseline_ratio_lag1'] = df.groupby('Province')['baseline_ratio'].shift(1)

    # Binary regime indicator based on recent performance
    # If last 3 months averaged <50% of baseline, we're in a "low" regime
    df['recent_low_regime'] = (df['cases_rolling_3m'] < df['hist_mean'] * 0.5).astype(int)

    return df


def build_regime_aware_model(df: pd.DataFrame):
    """
    Build a model that includes regime indicators.
    This allows the model to adjust predictions based on whether
    we're in a "normal" or "intervention" period.
    """
    df = detect_regime_changes(df)
    df = create_regime_features(df)

    # Drop rows with NA from feature engineering
    df = df.dropna(subset=['cases_rolling_3m', 'baseline_ratio_lag1'])

    # Time split
    train = df[df['Year'] < 2024].copy()
    test = df[df['Year'] == 2024].copy()

    print(f"\nTrain: {len(train)} rows, Test: {len(test)} rows")

    # Features
    # Climate features (what we can forecast)
    climate_features = ['iosd_lag1', 'iosd_lag2', 'precip_lag1']

    # Regime features (captures structural state)
    regime_features = ['baseline_ratio_lag1', 'recent_low_regime']

    # Categorical
    categorical_features = ['Season', 'Province']

    all_numeric = climate_features + regime_features

    print(f"\nFeatures:")
    print(f"  Climate: {climate_features}")
    print(f"  Regime:  {regime_features}")
    print(f"  Categorical: {categorical_features}")

    # Build pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), all_numeric),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))  # Ridge for stability
    ])

    # Prepare data
    X_train = train[all_numeric + categorical_features]
    y_train = train['malaria_cases']
    X_test = test[all_numeric + categorical_features]
    y_test = test['malaria_cases']

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Ensure non-negative predictions
    y_train_pred = np.maximum(y_train_pred, 0)
    y_test_pred = np.maximum(y_test_pred, 0)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    print(f"\n{'='*60}")
    print("REGIME-AWARE MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<15} {'Train':<15} {'Test':<15}")
    print(f"  {'-'*45}")
    print(f"  {'R²':<15} {train_r2:<15.4f} {test_r2:<15.4f}")
    print(f"  {'RMSE':<15} {np.sqrt(mean_squared_error(y_train, y_train_pred)):<15,.0f} {test_rmse:<15,.0f}")
    print(f"  {'MAE':<15} {mean_absolute_error(y_train, y_train_pred):<15,.0f} {test_mae:<15,.0f}")
    print(f"  {'MAPE':<15} {'-':<15} {test_mape:<15.1f}%")

    # Coefficients
    model = pipeline.named_steps['regressor']
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_features = all_numeric + cat_names

    print(f"\n  Coefficients (sorted by magnitude):")
    coef_df = pd.DataFrame({'feature': all_features, 'coef': model.coef_})
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    for _, row in coef_df.iterrows():
        print(f"    {row['feature']:<25} {row['coef']:>12,.2f}")
    print(f"    {'intercept':<25} {model.intercept_:>12,.2f}")

    # Results by province
    results = test[['Province', 'Year', 'Month', 'month_english', 'Season',
                    'malaria_cases', 'recent_low_regime']].copy()
    results['predicted'] = y_test_pred.astype(int)
    results['error'] = results['malaria_cases'] - results['predicted']
    results['error_pct'] = (results['error'] / results['malaria_cases'] * 100).round(1)

    print(f"\n{'='*60}")
    print("2024 PREDICTIONS BY PROVINCE")
    print(f"{'='*60}")

    for province in ['Gaza', 'Inhambane', 'Maputo']:
        prov_results = results[results['Province'] == province]
        prov_r2 = r2_score(prov_results['malaria_cases'], prov_results['predicted'])
        prov_mape = np.mean(np.abs(prov_results['error'] / prov_results['malaria_cases'])) * 100

        print(f"\n{province}:")
        print(f"  Test R²: {prov_r2:.4f}, MAPE: {prov_mape:.1f}%")
        print(f"  Low regime months: {prov_results['recent_low_regime'].sum()}/12")
        print(prov_results[['Month', 'Season', 'malaria_cases', 'predicted', 'error_pct']].head(6).to_string(index=False))

    # Summary comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"""
    Model                           Test R²    Test MAPE
    ------------------------------------------------
    Pooled (no regime)              0.32       198%
    Province-specific (no regime)   0.30       186%
    Season-interaction              0.27       191%
    REGIME-AWARE MODEL              {test_r2:.2f}       {test_mape:.0f}%

    The regime-aware model uses lagged baseline ratios to detect
    when a province has entered a "low transmission" regime,
    allowing it to adjust predictions accordingly.
    """)

    results.to_csv('predictions_regime_aware_2024.csv', index=False)
    print("Saved: predictions_regime_aware_2024.csv")

    return pipeline, results


def demonstrate_operational_use(df: pd.DataFrame, pipeline):
    """
    Show how to use the model operationally.
    """
    print(f"\n{'='*60}")
    print("OPERATIONAL USE: How to Make Predictions")
    print(f"{'='*60}")
    print("""
    To predict next month's cases, you need:

    1. CLIMATE DATA (can be forecasted):
       - iosd_lag1: IOSD index from 1 month ago
       - iosd_lag2: IOSD index from 2 months ago
       - precip_lag1: Precipitation from 1 month ago

    2. REGIME INDICATORS (from recent data):
       - baseline_ratio_lag1: Last month's cases / historical mean
       - recent_low_regime: 1 if recent 3-month avg < 50% of baseline

    3. CATEGORICAL:
       - Season: 'Wet' or 'Dry'
       - Province: 'Gaza', 'Inhambane', or 'Maputo'

    EXAMPLE for Gaza, January 2025:
    - If December 2024 had 6,484 cases (hist mean ~30,000)
    - baseline_ratio_lag1 = 6484/30000 = 0.22
    - recent_low_regime = 1 (because recent avg << 50% baseline)
    - Model will predict LOW cases, accounting for intervention
    """)


def main():
    print("="*60)
    print("MALARIA EARLY WARNING SYSTEM")
    print("Regime Change Model")
    print("="*60)

    df = load_data('master_climate_malaria_provinces_2020_2024.csv')

    pipeline, results = build_regime_aware_model(df)

    demonstrate_operational_use(df, pipeline)

    return pipeline, results


if __name__ == "__main__":
    pipeline, results = main()
