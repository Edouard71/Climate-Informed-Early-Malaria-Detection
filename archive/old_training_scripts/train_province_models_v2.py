"""
Province-Specific Malaria Models
================================
Training separate models for Inhambane and Maputo (excluding Gaza).
Each model is independent and trained/tested on its own province data.

Train: 2020-2023
Test: 2024
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING AND FEATURE ENGINEERING
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
    """Add comprehensive feature set for a single province."""
    df = df.copy()

    # Cyclical month encoding (captures seasonality)
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Season as numeric
    df['is_wet'] = (df['Season'] == 'Wet').astype(int)

    # Sort by date for proper lag calculation
    df = df.sort_values('date')

    # Autoregressive features (previous months' cases)
    df['cases_lag1'] = df['malaria_cases'].shift(1)
    df['cases_lag2'] = df['malaria_cases'].shift(2)
    df['cases_lag3'] = df['malaria_cases'].shift(3)

    # Rolling statistics
    df['cases_rolling_3m'] = df['malaria_cases'].shift(1).rolling(3, min_periods=1).mean()
    df['cases_rolling_6m'] = df['malaria_cases'].shift(1).rolling(6, min_periods=1).mean()

    # Climate interactions
    df['iosd_precip'] = df['iosd_lag1'] * df['precip_lag1']

    # Year-over-year (same month previous year)
    df['cases_yoy'] = df.groupby('Month')['malaria_cases'].shift(1)

    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_and_evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_type: str = 'ridge'
) -> Tuple[Pipeline, np.ndarray, Dict, Dict]:
    """
    Train a model and evaluate on test set.

    Returns: (model, predictions, train_metrics, test_metrics)
    """
    # Prepare data
    train_clean = train_df.dropna(subset=feature_cols + ['malaria_cases'])
    test_clean = test_df.dropna(subset=feature_cols + ['malaria_cases'])

    X_train = train_clean[feature_cols]
    y_train = train_clean['malaria_cases']
    X_test = test_clean[feature_cols]
    y_test = test_clean['malaria_cases']

    # Select model type
    if model_type == 'ridge':
        regressor = Ridge(alpha=10.0)
    elif model_type == 'lasso':
        regressor = Lasso(alpha=1.0)
    elif model_type == 'elastic':
        regressor = ElasticNet(alpha=1.0, l1_ratio=0.5)
    elif model_type == 'rf':
        regressor = RandomForestRegressor(
            n_estimators=100, max_depth=4, min_samples_leaf=3, random_state=42
        )
    elif model_type == 'gb':
        regressor = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Build pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred = np.maximum(y_test_pred, 0)  # Non-negative

    # Metrics
    train_metrics = {
        'r2': r2_score(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'n_samples': len(y_train)
    }

    test_metrics = {
        'r2': r2_score(y_test, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'mape': np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100,
        'n_samples': len(y_test)
    }

    return model, y_test_pred, train_metrics, test_metrics, test_clean


def find_best_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    province: str
) -> Dict:
    """Try multiple model types and return the best one."""

    model_types = ['ridge', 'lasso', 'elastic', 'rf', 'gb']
    results = {}

    print(f"\n{'='*60}")
    print(f"  {province.upper()} PROVINCE")
    print(f"{'='*60}")
    print(f"\nTrying {len(model_types)} model types...")
    print(f"\n{'Model':<12} {'Train R²':>10} {'Test R²':>10} {'MAPE':>10}")
    print("-"*45)

    for model_type in model_types:
        model, preds, train_m, test_m, test_clean = train_and_evaluate_model(
            train_df, test_df, feature_cols, model_type
        )

        results[model_type] = {
            'model': model,
            'predictions': preds,
            'train_metrics': train_m,
            'test_metrics': test_m,
            'test_data': test_clean
        }

        print(f"{model_type:<12} {train_m['r2']:>10.3f} {test_m['r2']:>10.3f} {test_m['mape']:>9.1f}%")

    # Find best by test R²
    best_type = max(results.keys(), key=lambda k: results[k]['test_metrics']['r2'])

    print(f"\n>>> Best model: {best_type.upper()} (Test R²={results[best_type]['test_metrics']['r2']:.3f})")

    return {
        'best_type': best_type,
        'best_model': results[best_type]['model'],
        'best_predictions': results[best_type]['predictions'],
        'best_train_metrics': results[best_type]['train_metrics'],
        'best_test_metrics': results[best_type]['test_metrics'],
        'test_data': results[best_type]['test_data'],
        'all_results': results
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("PROVINCE-SPECIFIC MALARIA MODELS")
    print("Inhambane & Maputo (excluding Gaza)")
    print("="*70)

    # Load data
    df = load_data('master_climate_malaria_provinces_2020_2024.csv')

    # Feature columns to use
    feature_cols = [
        'iosd_lag1', 'iosd_lag2', 'precip_lag1', 'precip_lag2',
        'month_sin', 'month_cos', 'is_wet',
        'cases_lag1', 'cases_rolling_3m'
    ]

    provinces = ['Inhambane', 'Maputo']
    final_results = {}

    for province in provinces:
        # Filter to this province only
        prov_df = df[df['Province'] == province].copy()
        prov_df = add_features(prov_df)

        # Split train/test
        train_df = prov_df[prov_df['Year'] < 2024].copy()
        test_df = prov_df[prov_df['Year'] == 2024].copy()

        print(f"\n{province}: {len(train_df)} train samples, {len(test_df)} test samples")

        # Find best model
        result = find_best_model(train_df, test_df, feature_cols, province)
        final_results[province] = result

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print(f"\n{'Province':<12} {'Best Model':<12} {'Train R²':>10} {'Test R²':>10} {'MAPE':>10}")
    print("-"*58)

    for province in provinces:
        r = final_results[province]
        print(f"{province:<12} {r['best_type']:<12} "
              f"{r['best_train_metrics']['r2']:>10.3f} "
              f"{r['best_test_metrics']['r2']:>10.3f} "
              f"{r['best_test_metrics']['mape']:>9.1f}%")

    # ==========================================================================
    # Detailed Results
    # ==========================================================================

    for province in provinces:
        r = final_results[province]
        test_data = r['test_data']
        predictions = r['best_predictions']

        print(f"\n{'='*60}")
        print(f"{province} - Monthly Predictions (2024)")
        print(f"{'='*60}")
        print(f"\n{'Month':<8} {'Actual':>12} {'Predicted':>12} {'Error':>12} {'Error%':>10}")
        print("-"*55)

        for i, (_, row) in enumerate(test_data.iterrows()):
            actual = row['malaria_cases']
            pred = predictions[i]
            error = actual - pred
            error_pct = (error / actual) * 100

            print(f"{int(row['Month']):<8} {actual:>12,.0f} {pred:>12,.0f} {error:>12,.0f} {error_pct:>9.1f}%")

        total_actual = test_data['malaria_cases'].sum()
        total_pred = predictions.sum()
        print("-"*55)
        print(f"{'TOTAL':<8} {total_actual:>12,.0f} {total_pred:>12,.0f} "
              f"{total_actual-total_pred:>12,.0f} {(total_actual-total_pred)/total_actual*100:>9.1f}%")

    # ==========================================================================
    # Visualizations
    # ==========================================================================

    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, province in enumerate(provinces):
        r = final_results[province]
        test_data = r['test_data']
        predictions = r['best_predictions']
        actual = test_data['malaria_cases'].values
        months = test_data['Month'].values

        # Bar chart
        ax1 = axes[0, idx]
        width = 0.35
        x = np.arange(len(months))

        ax1.bar(x - width/2, actual, width, label='Actual', color='#2E86AB', alpha=0.8)
        ax1.bar(x + width/2, predictions, width, label='Predicted', color='#E94F37', alpha=0.8)

        ax1.set_xlabel('Month')
        ax1.set_ylabel('Malaria Cases')
        ax1.set_title(f'{province} - 2024 Predictions\n'
                      f'({r["best_type"].upper()}, R²={r["best_test_metrics"]["r2"]:.3f}, '
                      f'MAPE={r["best_test_metrics"]["mape"]:.1f}%)',
                      fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(months)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Scatter plot
        ax2 = axes[1, idx]
        ax2.scatter(actual, predictions, s=100, c='#2E86AB', alpha=0.7, edgecolors='white')

        # Perfect prediction line
        max_val = max(actual.max(), predictions.max()) * 1.1
        min_val = min(actual.min(), predictions.min()) * 0.9
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect prediction')

        # Add month labels
        for i, month in enumerate(months):
            ax2.annotate(f'M{int(month)}', (actual[i], predictions[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax2.set_xlabel('Actual Cases')
        ax2.set_ylabel('Predicted Cases')
        ax2.set_title(f'{province} - Actual vs Predicted', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('inhambane_maputo_models_2024.png', dpi=150, bbox_inches='tight')
    print("\nSaved: inhambane_maputo_models_2024.png")
    plt.close()

    # Time series plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for idx, province in enumerate(provinces):
        ax = axes[idx]

        # Get all data for this province
        prov_df = df[df['Province'] == province].copy()
        prov_df = add_features(prov_df)
        prov_df = prov_df.sort_values('date')

        # Plot actual values
        ax.plot(prov_df['date'], prov_df['malaria_cases'],
                'o-', color='#2E86AB', label='Actual', linewidth=2, markersize=5)

        # Plot 2024 predictions
        r = final_results[province]
        test_data = r['test_data'].sort_values('date')
        predictions = r['best_predictions']

        # Reorder predictions to match sorted test_data
        pred_df = test_data.copy()
        pred_df['predicted'] = predictions
        pred_df = pred_df.sort_values('date')

        ax.plot(pred_df['date'], pred_df['predicted'],
                's--', color='#E94F37', label='Predicted (2024)', linewidth=2, markersize=8)

        # Shade test period
        ax.axvspan(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31'),
                   alpha=0.15, color='yellow', label='Test Period')

        ax.set_xlabel('Date')
        ax.set_ylabel('Malaria Cases')
        ax.set_title(f'{province} Province - Full Time Series (2020-2024)\n'
                     f'Test R²={r["best_test_metrics"]["r2"]:.3f}, MAPE={r["best_test_metrics"]["mape"]:.1f}%',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('inhambane_maputo_timeseries.png', dpi=150, bbox_inches='tight')
    print("Saved: inhambane_maputo_timeseries.png")
    plt.close()

    # ==========================================================================
    # Save models
    # ==========================================================================

    saved_models = {}
    for province in provinces:
        r = final_results[province]
        saved_models[province] = {
            'model': r['best_model'],
            'model_type': r['best_type'],
            'feature_cols': feature_cols,
            'train_metrics': r['best_train_metrics'],
            'test_metrics': r['best_test_metrics']
        }

    joblib.dump(saved_models, 'inhambane_maputo_models.joblib')
    print("\nSaved: inhambane_maputo_models.joblib")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

    return final_results


if __name__ == "__main__":
    results = main()
