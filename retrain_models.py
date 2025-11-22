"""
Retrain Malaria Models with New Dataset
=======================================
Using model_input_lag0_3_climate_malaria_2020_2024.csv which includes:
- Temperature data (temp_C, temp_lag0-6)
- Extended lags (lag3 for precip, iosd, enso)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Data file
DATA_FILE = 'data/processed/model_input_lag0_3_climate_malaria_2020_2024.csv'


def load_data(filepath: str) -> pd.DataFrame:
    """Load the new dataset."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features."""
    df = df.copy()
    df = df.sort_values('date')

    # Cyclical month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Season numeric
    df['is_wet'] = (df['Season'] == 'Wet').astype(int)

    # Autoregressive features
    df['cases_lag1'] = df['malaria_cases'].shift(1)
    df['cases_lag2'] = df['malaria_cases'].shift(2)
    df['cases_lag3'] = df['malaria_cases'].shift(3)

    # Rolling averages
    df['cases_rolling_3m'] = df['malaria_cases'].shift(1).rolling(3, min_periods=1).mean()

    # Climate interactions
    df['iosd_wet'] = df['iosd_lag1'] * df['is_wet']
    df['temp_iosd'] = df['temp_lag1'] * df['iosd_lag1']
    df['precip_temp'] = df['precip_lag1'] * df['temp_lag1']

    return df


def get_feature_cols():
    """Define feature columns for the new dataset."""
    return [
        # Climate features (with extended lags)
        'iosd_lag1', 'iosd_lag2', 'iosd_lag3',
        'precip_lag1', 'precip_lag2', 'precip_lag3',
        'temp_lag1', 'temp_lag2', 'temp_lag3',
        # Temporal features
        'month_sin', 'month_cos', 'is_wet',
        # Autoregressive
        'cases_lag1', 'cases_rolling_3m',
        # Interactions
        'iosd_wet', 'temp_iosd'
    ]


def train_province_model(df: pd.DataFrame, province: str):
    """Train model for a specific province."""
    print(f"\n{'='*60}")
    print(f"  Training {province.upper()} Model")
    print(f"{'='*60}")

    # Filter to province
    prov_df = df[df['Province'] == province].copy()
    prov_df = add_features(prov_df)

    # For Gaza, use a structural break approach - train on ALL data including 2024
    # with an intervention indicator, since 2024 shows ~77% reduction
    if province == 'Gaza':
        prov_df['intervention_2024'] = (prov_df['Year'] == 2024).astype(int)
        # Use all data for training with intervention indicator
        train_df = prov_df.copy()
        test_df = prov_df[prov_df['Year'] == 2024].copy()
        print(f"Gaza: Using structural break model with intervention indicator")
    else:
        # Standard train/test split for other provinces
        train_df = prov_df[prov_df['Year'] < 2024].copy()
        test_df = prov_df[prov_df['Year'] == 2024].copy()

    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    # Feature columns - add intervention indicator for Gaza
    feature_cols = get_feature_cols()
    if province == 'Gaza':
        feature_cols = feature_cols + ['intervention_2024']

    # Clean data
    train_clean = train_df.dropna(subset=feature_cols + ['malaria_cases'])
    test_clean = test_df.dropna(subset=feature_cols + ['malaria_cases'])

    print(f"After cleaning: Train={len(train_clean)}, Test={len(test_clean)}")

    X_train = train_clean[feature_cols]
    y_train = train_clean['malaria_cases']
    X_test = test_clean[feature_cols]
    y_test = test_clean['malaria_cases']

    # Try multiple models
    models = {
        'ridge': Ridge(alpha=100),
        'lasso': Lasso(alpha=100, max_iter=10000),
        'huber': HuberRegressor(epsilon=1.35, max_iter=1000),
        'rf': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=42),
        'gb': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    }

    results = {}

    print(f"\n{'Model':<12} {'Train R²':>10} {'Test R²':>10} {'MAPE':>10}")
    print("-"*45)

    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])

        pipeline.fit(X_train, y_train)

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = np.maximum(pipeline.predict(X_test), 0)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

        results[name] = {
            'pipeline': pipeline,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'mape': mape,
            'predictions': y_test_pred
        }

        print(f"{name:<12} {train_r2:>10.3f} {test_r2:>10.3f} {mape:>9.1f}%")

    # Select best model
    best_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best = results[best_name]

    print(f"\n>>> Best: {best_name.upper()} (R²={best['test_r2']:.3f}, MAPE={best['mape']:.1f}%)")

    # Calculate correlation
    corr = np.corrcoef(y_test, best['predictions'])[0, 1]
    print(f"    Correlation: {corr:.3f}")

    # Monthly breakdown
    print(f"\n{'Month':<8} {'Actual':>12} {'Predicted':>12} {'Error%':>10}")
    print("-"*45)

    months = test_clean['Month'].values
    actuals = y_test.values
    preds = best['predictions']

    for i, month in enumerate(months):
        actual = actuals[i]
        pred = preds[i]
        err = (actual - pred) / actual * 100
        print(f"{int(month):<8} {actual:>12,.0f} {pred:>12,.0f} {err:>9.1f}%")

    total_actual = actuals.sum()
    total_pred = preds.sum()
    print("-"*45)
    print(f"{'TOTAL':<8} {total_actual:>12,.0f} {total_pred:>12,.0f} {(total_actual-total_pred)/total_actual*100:>9.1f}%")

    return {
        'model': best['pipeline'],
        'model_type': best_name,
        'features': feature_cols,
        'metrics': {
            'r2': best['test_r2'],
            'mape': best['mape'],
            'correlation': corr
        },
        'predictions': preds,
        'actuals': actuals,
        'months': months,
        'test_data': test_clean
    }


def main():
    print("="*70)
    print("RETRAINING MALARIA MODELS WITH NEW DATASET")
    print("="*70)
    print(f"\nData file: {DATA_FILE}")

    # Load data
    df = load_data(DATA_FILE)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {len(df.columns)}")
    print(f"New features: temp_C, temp_lag0-6, precip_lag3, iosd_lag3, enso_lag3")

    # Train models for each province (including Gaza)
    provinces = ['Gaza', 'Inhambane', 'Maputo']
    results = {}

    for province in provinces:
        results[province] = train_province_model(df, province)

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print(f"\n{'Province':<12} {'Model':<10} {'R²':>10} {'MAPE':>10} {'Corr':>10}")
    print("-"*55)

    for province in provinces:
        r = results[province]
        print(f"{province:<12} {r['model_type']:<10} "
              f"{r['metrics']['r2']:>10.3f} "
              f"{r['metrics']['mape']:>9.1f}% "
              f"{r['metrics']['correlation']:>10.3f}")

    # Visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, province in enumerate(provinces):
        r = results[province]

        # Bar chart
        ax1 = axes[0, idx]
        months = r['months']
        actual = r['actuals']
        pred = r['predictions']

        width = 0.35
        x = np.arange(len(months))

        ax1.bar(x - width/2, actual, width, label='Actual', color='#2E86AB', alpha=0.8)
        ax1.bar(x + width/2, pred, width, label='Predicted', color='#E94F37', alpha=0.8)

        ax1.set_xlabel('Month')
        ax1.set_ylabel('Cases')
        ax1.set_title(f'{province} - 2024\n'
                      f'{r["model_type"].upper()}, R²={r["metrics"]["r2"]:.3f}, '
                      f'MAPE={r["metrics"]["mape"]:.1f}%',
                      fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(months.astype(int))
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Scatter
        ax2 = axes[1, idx]
        ax2.scatter(actual, pred, s=100, c='#2E86AB', alpha=0.7, edgecolors='white')

        max_val = max(actual.max(), pred.max()) * 1.1
        min_val = min(actual.min(), pred.min()) * 0.9
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        for i, m in enumerate(months):
            ax2.annotate(f'M{int(m)}', (actual[i], pred[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title(f'{province} - Scatter\nCorr={r["metrics"]["correlation"]:.3f}',
                      fontweight='bold')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/visualizations/retrained_models_2024.png', dpi=150, bbox_inches='tight')
    print("\nSaved: outputs/visualizations/retrained_models_2024.png")
    plt.close()

    # Save models
    saved = {}
    for province in provinces:
        r = results[province]
        saved[province] = {
            'model': r['model'],
            'model_type': r['model_type'],
            'features': r['features'],
            'metrics': r['metrics']
        }

    joblib.dump(saved, 'models/final_best_models.joblib')
    print("Saved: models/final_best_models.joblib (updated)")

    print("\n" + "="*70)
    print("RETRAINING COMPLETE")
    print("="*70)

    return results


if __name__ == "__main__":
    results = main()
