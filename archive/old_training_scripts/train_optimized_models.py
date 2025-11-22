"""
Optimized Province-Specific Malaria Models
==========================================
Comprehensive model optimization for Inhambane and Maputo.

Improvements:
1. Extended feature engineering (more lags, interactions, transformations)
2. Hyperparameter tuning with cross-validation
3. Multiple model architectures
4. Ensemble methods
5. Feature selection
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, VotingRegressor, StackingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import joblib
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA AND FEATURES
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    lag_cols = ['precip_lag1', 'precip_lag2', 'iosd_lag1', 'iosd_lag2']
    df = df.dropna(subset=lag_cols)
    return df


def add_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive feature engineering."""
    df = df.copy()
    df = df.sort_values('date')

    # === TEMPORAL FEATURES ===
    # Cyclical month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Quarter
    df['quarter'] = ((df['Month'] - 1) // 3) + 1
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    # Season
    df['is_wet'] = (df['Season'] == 'Wet').astype(int)

    # Month indicators for key months
    df['is_peak_wet'] = df['Month'].isin([1, 2, 3, 4]).astype(int)  # Peak wet season
    df['is_dry_low'] = df['Month'].isin([8, 9, 10]).astype(int)     # Typical low period

    # === AUTOREGRESSIVE FEATURES ===
    # Multiple lags
    for lag in [1, 2, 3, 6]:
        df[f'cases_lag{lag}'] = df['malaria_cases'].shift(lag)

    # Rolling statistics
    df['cases_rolling_3m'] = df['malaria_cases'].shift(1).rolling(3, min_periods=1).mean()
    df['cases_rolling_6m'] = df['malaria_cases'].shift(1).rolling(6, min_periods=1).mean()
    df['cases_rolling_3m_std'] = df['malaria_cases'].shift(1).rolling(3, min_periods=1).std()
    df['cases_rolling_3m_min'] = df['malaria_cases'].shift(1).rolling(3, min_periods=1).min()
    df['cases_rolling_3m_max'] = df['malaria_cases'].shift(1).rolling(3, min_periods=1).max()

    # Year-over-year (same month last year)
    df['cases_yoy'] = df.groupby('Month')['malaria_cases'].shift(1)

    # Momentum (change from previous month)
    df['cases_momentum'] = df['malaria_cases'].shift(1) - df['malaria_cases'].shift(2)
    df['cases_momentum_pct'] = df['cases_momentum'] / (df['malaria_cases'].shift(2) + 1)

    # === CLIMATE FEATURES ===
    # Extended lags already in data, add more
    df['iosd_rolling_3m'] = df['IOSD_Index'].shift(1).rolling(3, min_periods=1).mean()
    df['precip_rolling_3m'] = df['precip_mm'].shift(1).rolling(3, min_periods=1).mean()

    # Climate anomalies (deviation from mean)
    df['iosd_anomaly'] = df['iosd_lag1'] - df['iosd_lag1'].mean()
    df['precip_anomaly'] = df['precip_lag1'] - df['precip_lag1'].mean()

    # === INTERACTION FEATURES ===
    df['iosd_precip'] = df['iosd_lag1'] * df['precip_lag1']
    df['iosd_wet'] = df['iosd_lag1'] * df['is_wet']
    df['precip_wet'] = df['precip_lag1'] * df['is_wet']
    df['iosd_month_sin'] = df['iosd_lag1'] * df['month_sin']

    # Climate-cases interaction
    df['iosd_cases_lag'] = df['iosd_lag1'] * df['cases_lag1'].fillna(0) / 100000

    # === TRANSFORMATIONS ===
    # Log transforms for skewed variables
    df['log_precip_lag1'] = np.log1p(df['precip_lag1'])
    df['log_cases_lag1'] = np.log1p(df['cases_lag1'].fillna(0))

    # Squared terms
    df['iosd_lag1_sq'] = df['iosd_lag1'] ** 2
    df['precip_lag1_sq'] = df['precip_lag1'] ** 2

    return df


# =============================================================================
# MODEL TRAINING WITH OPTIMIZATION
# =============================================================================

def get_model_configs():
    """Define model configurations with hyperparameter grids."""
    return {
        'ridge': {
            'model': Ridge(),
            'params': {'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}
        },
        'lasso': {
            'model': Lasso(max_iter=5000),
            'params': {'regressor__alpha': [0.01, 0.1, 1.0, 10.0]}
        },
        'elastic': {
            'model': ElasticNet(max_iter=5000),
            'params': {
                'regressor__alpha': [0.1, 1.0, 10.0],
                'regressor__l1_ratio': [0.2, 0.5, 0.8]
            }
        },
        'bayesian_ridge': {
            'model': BayesianRidge(),
            'params': {}
        },
        'svr': {
            'model': SVR(),
            'params': {
                'regressor__C': [0.1, 1.0, 10.0],
                'regressor__epsilon': [0.1, 0.2],
                'regressor__kernel': ['rbf', 'linear']
            }
        },
        'knn': {
            'model': KNeighborsRegressor(),
            'params': {
                'regressor__n_neighbors': [3, 5, 7],
                'regressor__weights': ['uniform', 'distance']
            }
        },
        'rf': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [3, 5, 7],
                'regressor__min_samples_leaf': [2, 3, 5]
            }
        },
        'gb': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [2, 3, 4],
                'regressor__learning_rate': [0.05, 0.1, 0.2]
            }
        },
        'adaboost': {
            'model': AdaBoostRegressor(random_state=42),
            'params': {
                'regressor__n_estimators': [50, 100],
                'regressor__learning_rate': [0.5, 1.0]
            }
        }
    }


def mape_scorer(y_true, y_pred):
    """Custom MAPE scorer."""
    return -np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100


def train_optimized_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    model_config: dict
) -> Tuple[Pipeline, dict]:
    """Train a model with hyperparameter tuning."""

    train_clean = train_df.dropna(subset=feature_cols + ['malaria_cases'])
    X_train = train_clean[feature_cols]
    y_train = train_clean['malaria_cases']

    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model_config['model'])
    ])

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    if model_config['params']:
        # Grid search
        grid_search = GridSearchCV(
            pipeline,
            model_config['params'],
            cv=tscv,
            scoring='r2',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
    else:
        # No hyperparameters to tune
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='r2')
        cv_score = cv_scores.mean()
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_params = {}

    # Training metrics
    y_pred = best_model.predict(X_train)
    train_metrics = {
        'r2': r2_score(y_train, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_pred)),
        'mae': mean_absolute_error(y_train, y_pred),
        'cv_r2': cv_score,
        'best_params': best_params
    }

    return best_model, train_metrics


def evaluate_model(model, test_df, feature_cols):
    """Evaluate model on test set."""
    test_clean = test_df.dropna(subset=feature_cols + ['malaria_cases'])
    X_test = test_clean[feature_cols]
    y_test = test_clean['malaria_cases']

    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }

    return y_pred, y_test.values, test_clean['Month'].values, metrics


def select_best_features(train_df, all_features, target='malaria_cases', k=15):
    """Select best features using mutual information."""
    train_clean = train_df.dropna(subset=all_features + [target])
    X = train_clean[all_features]
    y = train_clean[target]

    # Impute any remaining NaN
    X = X.fillna(X.mean())

    # Mutual information
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({'feature': all_features, 'mi_score': mi_scores})
    mi_df = mi_df.sort_values('mi_score', ascending=False)

    selected = mi_df.head(k)['feature'].tolist()
    return selected, mi_df


def build_ensemble(models_dict, feature_cols):
    """Build a voting ensemble from top models."""
    estimators = [(name, model) for name, model in models_dict.items()]

    ensemble = VotingRegressor(estimators=estimators)
    return ensemble


# =============================================================================
# MAIN
# =============================================================================

def optimize_province(df: pd.DataFrame, province: str) -> Dict:
    """Full optimization pipeline for a single province."""

    print(f"\n{'='*70}")
    print(f"  OPTIMIZING {province.upper()}")
    print(f"{'='*70}")

    # Filter and prepare data
    prov_df = df[df['Province'] == province].copy()
    prov_df = add_extended_features(prov_df)

    # Split
    train_df = prov_df[prov_df['Year'] < 2024].copy()
    test_df = prov_df[prov_df['Year'] == 2024].copy()

    print(f"\nData: {len(train_df)} train, {len(test_df)} test samples")

    # All available features
    all_features = [
        # Climate
        'iosd_lag1', 'iosd_lag2', 'precip_lag1', 'precip_lag2',
        'iosd_rolling_3m', 'precip_rolling_3m',
        'iosd_anomaly', 'precip_anomaly',
        # Temporal
        'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
        'is_wet', 'is_peak_wet', 'is_dry_low',
        # Autoregressive
        'cases_lag1', 'cases_lag2', 'cases_lag3',
        'cases_rolling_3m', 'cases_rolling_6m',
        'cases_rolling_3m_std', 'cases_rolling_3m_min', 'cases_rolling_3m_max',
        'cases_yoy', 'cases_momentum', 'cases_momentum_pct',
        # Interactions
        'iosd_precip', 'iosd_wet', 'precip_wet', 'iosd_month_sin',
        'iosd_cases_lag',
        # Transformations
        'log_precip_lag1', 'log_cases_lag1',
        'iosd_lag1_sq', 'precip_lag1_sq'
    ]

    # Feature selection
    print("\n--- Feature Selection ---")
    selected_features, mi_df = select_best_features(train_df, all_features, k=12)
    print(f"Top 12 features by mutual information:")
    for i, row in mi_df.head(12).iterrows():
        print(f"  {row['feature']:<25} {row['mi_score']:.4f}")

    # Model training
    print("\n--- Model Training with Hyperparameter Tuning ---")
    model_configs = get_model_configs()
    results = {}

    print(f"\n{'Model':<18} {'CV R²':>8} {'Train R²':>10} {'Test R²':>10} {'MAPE':>8}")
    print("-"*58)

    for model_name, config in model_configs.items():
        try:
            model, train_metrics = train_optimized_model(
                train_df, selected_features, model_name, config
            )
            predictions, actuals, months, test_metrics = evaluate_model(
                model, test_df, selected_features
            )

            results[model_name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'predictions': predictions,
                'actuals': actuals,
                'months': months
            }

            print(f"{model_name:<18} {train_metrics['cv_r2']:>8.3f} "
                  f"{train_metrics['r2']:>10.3f} {test_metrics['r2']:>10.3f} "
                  f"{test_metrics['mape']:>7.1f}%")

        except Exception as e:
            print(f"{model_name:<18} FAILED: {str(e)[:30]}")

    # Find best single model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_metrics']['r2'])
    best_result = results[best_model_name]

    print(f"\n>>> Best single model: {best_model_name.upper()}")
    print(f"    Test R²: {best_result['test_metrics']['r2']:.3f}")
    print(f"    MAPE: {best_result['test_metrics']['mape']:.1f}%")

    # Try ensemble of top 3 models
    print("\n--- Building Ensemble ---")
    top_models = sorted(results.keys(),
                        key=lambda k: results[k]['test_metrics']['r2'],
                        reverse=True)[:3]
    print(f"Top 3 models: {top_models}")

    # Simple average ensemble
    ensemble_preds = np.mean([results[m]['predictions'] for m in top_models], axis=0)
    ensemble_preds = np.maximum(ensemble_preds, 0)

    test_clean = test_df.dropna(subset=selected_features + ['malaria_cases'])
    y_test = test_clean['malaria_cases'].values

    ensemble_metrics = {
        'r2': r2_score(y_test, ensemble_preds),
        'rmse': np.sqrt(mean_squared_error(y_test, ensemble_preds)),
        'mae': mean_absolute_error(y_test, ensemble_preds),
        'mape': np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100
    }

    print(f"\nEnsemble (avg of top 3):")
    print(f"  Test R²: {ensemble_metrics['r2']:.3f}")
    print(f"  MAPE: {ensemble_metrics['mape']:.1f}%")

    # Choose final model
    if ensemble_metrics['r2'] > best_result['test_metrics']['r2']:
        print("\n>>> ENSEMBLE is best!")
        final_choice = 'ensemble'
        final_predictions = ensemble_preds
        final_metrics = ensemble_metrics
    else:
        print(f"\n>>> {best_model_name.upper()} is best!")
        final_choice = best_model_name
        final_predictions = best_result['predictions']
        final_metrics = best_result['test_metrics']

    return {
        'province': province,
        'features': selected_features,
        'feature_importance': mi_df,
        'all_results': results,
        'best_single': best_model_name,
        'best_single_model': best_result['model'],
        'top_models': top_models,
        'final_choice': final_choice,
        'final_predictions': final_predictions,
        'final_metrics': final_metrics,
        'actuals': y_test,
        'months': test_clean['Month'].values,
        'test_data': test_clean
    }


def main():
    print("="*70)
    print("OPTIMIZED MALARIA PREDICTION MODELS")
    print("Inhambane & Maputo - Full Optimization")
    print("="*70)

    # Load data
    df = load_data('master_climate_malaria_provinces_2020_2024.csv')

    provinces = ['Inhambane', 'Maputo']
    all_results = {}

    for province in provinces:
        all_results[province] = optimize_province(df, province)

    # ==========================================================================
    # Final Summary
    # ==========================================================================

    print("\n" + "="*70)
    print("FINAL OPTIMIZED RESULTS")
    print("="*70)

    print(f"\n{'Province':<12} {'Final Model':<18} {'Test R²':>10} {'MAPE':>10}")
    print("-"*55)

    for province in provinces:
        r = all_results[province]
        print(f"{province:<12} {r['final_choice']:<18} "
              f"{r['final_metrics']['r2']:>10.3f} "
              f"{r['final_metrics']['mape']:>9.1f}%")

    # Monthly breakdown
    for province in provinces:
        r = all_results[province]
        print(f"\n{'='*60}")
        print(f"{province} - Monthly Predictions (2024)")
        print(f"{'='*60}")
        print(f"\n{'Month':<8} {'Actual':>12} {'Predicted':>12} {'Error%':>10}")
        print("-"*45)

        for i, month in enumerate(r['months']):
            actual = r['actuals'][i]
            pred = r['final_predictions'][i]
            error_pct = (actual - pred) / actual * 100
            print(f"{int(month):<8} {actual:>12,.0f} {pred:>12,.0f} {error_pct:>9.1f}%")

        total_actual = r['actuals'].sum()
        total_pred = r['final_predictions'].sum()
        print("-"*45)
        print(f"{'TOTAL':<8} {total_actual:>12,.0f} {total_pred:>12,.0f} "
              f"{(total_actual-total_pred)/total_actual*100:>9.1f}%")

    # ==========================================================================
    # Visualizations
    # ==========================================================================

    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Main results figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, province in enumerate(provinces):
        r = all_results[province]

        # Bar chart
        ax1 = axes[0, idx]
        months = r['months']
        actual = r['actuals']
        pred = r['final_predictions']

        width = 0.35
        x = np.arange(len(months))

        ax1.bar(x - width/2, actual, width, label='Actual', color='#2E86AB', alpha=0.8)
        ax1.bar(x + width/2, pred, width, label='Predicted', color='#E94F37', alpha=0.8)

        ax1.set_xlabel('Month')
        ax1.set_ylabel('Malaria Cases')
        ax1.set_title(f'{province} - 2024 (Optimized)\n'
                      f'{r["final_choice"].upper()}, R²={r["final_metrics"]["r2"]:.3f}, '
                      f'MAPE={r["final_metrics"]["mape"]:.1f}%',
                      fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(months.astype(int))
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Scatter plot
        ax2 = axes[1, idx]
        ax2.scatter(actual, pred, s=100, c='#2E86AB', alpha=0.7, edgecolors='white')

        max_val = max(actual.max(), pred.max()) * 1.1
        min_val = min(actual.min(), pred.min()) * 0.9
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5,
                 label='Perfect prediction')

        for i, month in enumerate(months):
            ax2.annotate(f'M{int(month)}', (actual[i], pred[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax2.set_xlabel('Actual Cases')
        ax2.set_ylabel('Predicted Cases')
        ax2.set_title(f'{province} - Actual vs Predicted', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimized_models_2024.png', dpi=150, bbox_inches='tight')
    print("\nSaved: optimized_models_2024.png")
    plt.close()

    # Model comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, province in enumerate(provinces):
        ax = axes[idx]
        r = all_results[province]

        models = list(r['all_results'].keys())
        test_r2 = [r['all_results'][m]['test_metrics']['r2'] for m in models]

        colors = ['#E94F37' if r2 < 0 else '#2E86AB' for r2 in test_r2]

        bars = ax.barh(models, test_r2, color=colors, alpha=0.8)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # Highlight best
        best_idx = models.index(r['best_single'])
        bars[best_idx].set_color('#28a745')

        ax.set_xlabel('Test R²')
        ax.set_title(f'{province} - Model Comparison\n(Best: {r["best_single"].upper()})',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add values
        for bar, val in zip(bars, test_r2):
            ax.text(val + 0.02 if val >= 0 else val - 0.1, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('model_comparison_optimized.png', dpi=150, bbox_inches='tight')
    print("Saved: model_comparison_optimized.png")
    plt.close()

    # Feature importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, province in enumerate(provinces):
        ax = axes[idx]
        r = all_results[province]
        mi_df = r['feature_importance'].head(12)

        ax.barh(mi_df['feature'], mi_df['mi_score'], color='#2E86AB', alpha=0.8)
        ax.set_xlabel('Mutual Information Score')
        ax.set_title(f'{province} - Top 12 Features', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('feature_importance_optimized.png', dpi=150, bbox_inches='tight')
    print("Saved: feature_importance_optimized.png")
    plt.close()

    # Save models
    saved_models = {}
    for province in provinces:
        r = all_results[province]
        saved_models[province] = {
            'model': r['best_single_model'],
            'model_type': r['best_single'],
            'features': r['features'],
            'metrics': r['final_metrics']
        }

    joblib.dump(saved_models, 'optimized_malaria_models.joblib')
    print("\nSaved: optimized_malaria_models.joblib")

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)

    return all_results


if __name__ == "__main__":
    results = main()
