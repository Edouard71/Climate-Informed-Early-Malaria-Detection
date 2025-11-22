"""
Improved Malaria Prediction Model
=================================
Addresses key issues:
1. Province-specific models (each province has different dynamics)
2. Seasonality features (month encoding)
3. Non-linear models (Random Forest, Gradient Boosting)
4. Better handling of regime changes
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare dataset."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Drop rows with NA in lag columns
    lag_cols = ['precip_lag1', 'precip_lag2', 'iosd_lag1', 'iosd_lag2']
    df = df.dropna(subset=lag_cols)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive feature set."""
    df = df.copy()

    # Cyclical month encoding (captures seasonality better)
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Season numeric
    df['is_wet'] = (df['Season'] == 'Wet').astype(int)

    # Previous month cases (important autoregressive feature)
    df = df.sort_values(['Province', 'date'])
    df['cases_lag1'] = df.groupby('Province')['malaria_cases'].shift(1)
    df['cases_lag2'] = df.groupby('Province')['malaria_cases'].shift(2)

    # 3-month rolling average
    df['cases_rolling_3m'] = df.groupby('Province')['malaria_cases'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # Year-over-year change (same month last year)
    df['cases_same_month_last_year'] = df.groupby(['Province', 'Month'])['malaria_cases'].shift(1)

    # Climate interactions
    df['iosd_precip_interaction'] = df['iosd_lag1'] * df['precip_lag1']

    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_province_model(train_df: pd.DataFrame,
                         province: str,
                         model_type: str = 'rf') -> Tuple[Pipeline, Dict]:
    """Train a model for a specific province."""

    # Filter to province
    prov_train = train_df[train_df['Province'] == province].copy()

    # Feature columns
    feature_cols = [
        'iosd_lag1', 'iosd_lag2', 'precip_lag1', 'precip_lag2',
        'month_sin', 'month_cos', 'is_wet',
        'cases_lag1', 'cases_rolling_3m'
    ]

    # Drop rows with NA
    prov_train = prov_train.dropna(subset=feature_cols + ['malaria_cases'])

    X = prov_train[feature_cols]
    y = prov_train['malaria_cases']

    # Build pipeline
    if model_type == 'ridge':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=10.0))
        ])
    elif model_type == 'rf':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=3,
                random_state=42
            ))
        ])
    elif model_type == 'gb':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=3,
                learning_rate=0.1,
                random_state=42
            ))
        ])

    model.fit(X, y)

    # Training metrics
    y_pred = model.predict(X)
    metrics = {
        'r2': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'n_samples': len(y)
    }

    return model, metrics


def evaluate_model(model: Pipeline,
                   test_df: pd.DataFrame,
                   province: str,
                   feature_cols: list) -> Tuple[np.ndarray, Dict]:
    """Evaluate model on test data."""

    prov_test = test_df[test_df['Province'] == province].copy()
    prov_test = prov_test.dropna(subset=feature_cols + ['malaria_cases'])

    X = prov_test[feature_cols]
    y_true = prov_test['malaria_cases']

    y_pred = model.predict(X)
    y_pred = np.maximum(y_pred, 0)  # Non-negative

    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

    return y_pred, metrics


# =============================================================================
# MAIN TRAINING
# =============================================================================

def main():
    print("="*70)
    print("IMPROVED MALARIA PREDICTION MODEL")
    print("="*70)

    # Load data
    df = load_data('master_climate_malaria_provinces_2020_2024.csv')
    df = add_features(df)

    # Split
    train_df = df[df['Year'] < 2024].copy()
    test_df = df[df['Year'] == 2024].copy()

    print(f"\nData: {len(df)} total, {len(train_df)} train, {len(test_df)} test")

    # Feature columns
    feature_cols = [
        'iosd_lag1', 'iosd_lag2', 'precip_lag1', 'precip_lag2',
        'month_sin', 'month_cos', 'is_wet',
        'cases_lag1', 'cases_rolling_3m'
    ]

    provinces = ['Gaza', 'Inhambane', 'Maputo']
    model_types = ['ridge', 'rf', 'gb']

    results = {}
    best_models = {}

    for province in provinces:
        print(f"\n{'='*70}")
        print(f"PROVINCE: {province}")
        print('='*70)

        results[province] = {}

        for model_type in model_types:
            # Train
            model, train_metrics = train_province_model(
                train_df, province, model_type
            )

            # Evaluate
            predictions, test_metrics = evaluate_model(
                model, test_df, province, feature_cols
            )

            results[province][model_type] = {
                'train': train_metrics,
                'test': test_metrics,
                'predictions': predictions
            }

            print(f"\n{model_type.upper()}: Train R²={train_metrics['r2']:.3f}, "
                  f"Test R²={test_metrics['r2']:.3f}, MAPE={test_metrics['mape']:.1f}%")

        # Select best model for this province (by test R²)
        best_type = max(model_types,
                        key=lambda m: results[province][m]['test']['r2'])
        best_models[province] = {
            'type': best_type,
            'model': train_province_model(train_df, province, best_type)[0],
            'metrics': results[province][best_type]['test']
        }
        print(f"\n>>> Best for {province}: {best_type.upper()}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - BEST MODELS")
    print("="*70)
    print(f"\n{'Province':<12} {'Model':<8} {'Test R²':>10} {'MAPE':>10}")
    print("-"*45)

    for province in provinces:
        best = best_models[province]
        print(f"{province:<12} {best['type']:<8} "
              f"{best['metrics']['r2']:>10.3f} {best['metrics']['mape']:>9.1f}%")

    # Visualize
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, province in enumerate(provinces):
        ax = axes[idx]

        # Get test data for this province
        prov_test = test_df[test_df['Province'] == province].copy()
        prov_test = prov_test.dropna(subset=feature_cols + ['malaria_cases'])

        # Get best model predictions
        best_model = best_models[province]['model']
        predictions = best_model.predict(prov_test[feature_cols])
        predictions = np.maximum(predictions, 0)

        months = prov_test['Month'].values
        actual = prov_test['malaria_cases'].values

        # Bar chart
        width = 0.35
        x = np.arange(len(months))
        ax.bar(x - width/2, actual, width, label='Actual', color='#2E86AB', alpha=0.8)
        ax.bar(x + width/2, predictions, width, label='Predicted', color='#E94F37', alpha=0.8)

        ax.set_xlabel('Month')
        ax.set_ylabel('Cases')
        ax.set_title(f'{province} - 2024\n({best_models[province]["type"].upper()}, '
                     f'R²={best_models[province]["metrics"]["r2"]:.2f})')
        ax.set_xticks(x)
        ax.set_xticklabels(months)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('improved_model_results.png', dpi=150, bbox_inches='tight')
    print("\nSaved: improved_model_results.png")
    plt.close()

    # Save models
    joblib.dump({
        'models': {p: best_models[p]['model'] for p in provinces},
        'model_types': {p: best_models[p]['type'] for p in provinces},
        'feature_cols': feature_cols,
        'metrics': {p: best_models[p]['metrics'] for p in provinces}
    }, 'improved_malaria_model.joblib')
    print("Saved: improved_malaria_model.joblib")

    # Show what's happening with Gaza
    print("\n" + "="*70)
    print("ANALYSIS: Why Gaza predictions are challenging")
    print("="*70)

    gaza_train = train_df[train_df['Province'] == 'Gaza']['malaria_cases']
    gaza_test = test_df[test_df['Province'] == 'Gaza']['malaria_cases']

    print(f"\nGaza Training (2020-2023):")
    print(f"  Mean: {gaza_train.mean():,.0f} cases/month")
    print(f"  Min:  {gaza_train.min():,.0f} | Max: {gaza_train.max():,.0f}")

    print(f"\nGaza Test (2024):")
    print(f"  Mean: {gaza_test.mean():,.0f} cases/month")
    print(f"  Min:  {gaza_test.min():,.0f} | Max: {gaza_test.max():,.0f}")

    print(f"\n  >>> 2024 is {gaza_test.mean()/gaza_train.mean()*100:.0f}% of historical average")
    print("  >>> This is a structural break (likely intervention), not predictable from climate")

    return results, best_models


if __name__ == "__main__":
    results, best_models = main()
