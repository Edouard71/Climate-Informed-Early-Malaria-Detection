"""
Malaria Early Warning System - Production Model
================================================
Climate-linked malaria prediction for Southern Mozambique.

This module provides:
1. MalariaPredictor class for training and inference
2. Functions for regime detection and feature engineering
3. Visualization utilities

Based on research showing:
- IOSD (Indian Ocean Subtropical Dipole) is strongest climate predictor
- Precipitation at 1-month lag contributes to predictions
- Regime indicators capture intervention effects

Author: Climate-Malaria Early Warning Project
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# CONSTANTS
# ==============================================================================

PROVINCES = ['Gaza', 'Inhambane', 'Maputo']

POPULATION_ESTIMATES = {
    'Maputo': 2_500_000,
    'Gaza': 1_450_000,
    'Inhambane': 1_560_000
}

# Historical baselines (2020-2023 monthly averages)
HISTORICAL_BASELINES = {
    'Gaza': {'mean': 32263, 'std': 9458},
    'Inhambane': {'mean': 65826, 'std': 17077},
    'Maputo': {'mean': 3980, 'std': 1818}
}

# Feature definitions
CLIMATE_FEATURES = ['iosd_lag1', 'iosd_lag2', 'precip_lag1']
REGIME_FEATURES = ['baseline_ratio_lag1', 'recent_low_regime']
CATEGORICAL_FEATURES = ['Season', 'Province']

# Alert thresholds (incidence per 1,000 population)
ALERT_THRESHOLDS = {
    'low': 2.0,
    'moderate': 3.5,
    'high': 5.0,
    'very_high': float('inf')
}


# ==============================================================================
# DATA PREPARATION
# ==============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and prepare malaria-climate dataset.

    Parameters
    ----------
    filepath : str
        Path to CSV file with columns: Province, Year, Month, date,
        malaria_cases, precip_mm, IOSD_Index, ENSO_ONI, and lag variables.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with date parsing and NA handling.
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Drop rows with NA in required lag columns
    lag_cols = ['precip_lag1', 'precip_lag2', 'iosd_lag1', 'iosd_lag2']
    df = df.dropna(subset=lag_cols)

    return df


def add_regime_features(df: pd.DataFrame,
                        historical_baselines: Optional[Dict] = None) -> pd.DataFrame:
    """
    Add regime detection features to dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with malaria_cases and Province columns.
    historical_baselines : dict, optional
        Dict of {province: {'mean': float, 'std': float}}.
        If None, uses HISTORICAL_BASELINES constant.

    Returns
    -------
    pd.DataFrame
        Dataset with added regime features.
    """
    df = df.copy()
    baselines = historical_baselines or HISTORICAL_BASELINES

    # Add historical baseline stats
    df['hist_mean'] = df['Province'].map(lambda p: baselines[p]['mean'])
    df['hist_std'] = df['Province'].map(lambda p: baselines[p]['std'])

    # Calculate baseline ratio
    df['baseline_ratio'] = df['malaria_cases'] / df['hist_mean']

    # Sort for proper lag calculation
    df = df.sort_values(['Province', 'date'])

    # Lagged baseline ratio
    df['baseline_ratio_lag1'] = df.groupby('Province')['baseline_ratio'].shift(1)

    # Rolling 3-month average
    df['cases_rolling_3m'] = df.groupby('Province')['malaria_cases'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # Low regime indicator (recent avg < 50% of baseline)
    df['recent_low_regime'] = (df['cases_rolling_3m'] < df['hist_mean'] * 0.5).astype(int)

    return df


# ==============================================================================
# MODEL CLASS
# ==============================================================================

class MalariaPredictor:
    """
    Regime-aware malaria prediction model.

    This model combines climate variables (IOSD, precipitation) with
    regime indicators to predict monthly malaria cases. It automatically
    detects intervention periods and adjusts predictions accordingly.

    Attributes
    ----------
    pipeline : sklearn.Pipeline
        Fitted prediction pipeline.
    historical_baselines : dict
        Province-level historical case statistics.
    metadata : dict
        Model training metadata.

    Example
    -------
    >>> model = MalariaPredictor()
    >>> model.fit(train_df)
    >>> predictions = model.predict(test_df)
    """

    def __init__(self, historical_baselines: Optional[Dict] = None):
        """
        Initialize predictor.

        Parameters
        ----------
        historical_baselines : dict, optional
            Custom historical baselines by province.
        """
        self.pipeline = None
        self.historical_baselines = historical_baselines or HISTORICAL_BASELINES
        self.metadata = {}

    def fit(self, df: pd.DataFrame) -> 'MalariaPredictor':
        """
        Train the model.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with required columns.

        Returns
        -------
        MalariaPredictor
            Fitted model instance.
        """
        # Add regime features
        df = add_regime_features(df, self.historical_baselines)
        df = df.dropna(subset=['baseline_ratio_lag1'])

        # Prepare features
        numeric_features = CLIMATE_FEATURES + REGIME_FEATURES

        X = df[numeric_features + CATEGORICAL_FEATURES]
        y = df['malaria_cases']

        # Build pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False),
                 CATEGORICAL_FEATURES)
            ]
        )

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Ridge(alpha=1.0))
        ])

        # Fit
        self.pipeline.fit(X, y)

        # Store metadata
        self.metadata = {
            'train_samples': len(df),
            'train_date_range': (df['date'].min(), df['date'].max()),
            'features': numeric_features + CATEGORICAL_FEATURES,
            'trained_at': datetime.now().isoformat()
        }

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        df : pd.DataFrame
            Data with required feature columns.

        Returns
        -------
        np.ndarray
            Predicted malaria cases.
        """
        if self.pipeline is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Add regime features if not present
        if 'baseline_ratio_lag1' not in df.columns:
            df = add_regime_features(df, self.historical_baselines)

        numeric_features = CLIMATE_FEATURES + REGIME_FEATURES
        X = df[numeric_features + CATEGORICAL_FEATURES]

        predictions = self.pipeline.predict(X)
        return np.maximum(predictions, 0)  # Ensure non-negative

    def evaluate(self, df: pd.DataFrame) -> Dict:
        """
        Evaluate model on dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Evaluation data with malaria_cases column.

        Returns
        -------
        dict
            Dictionary of evaluation metrics.
        """
        y_true = df['malaria_cases']
        y_pred = self.predict(df)

        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

    def save(self, filepath: str):
        """Save model to file."""
        joblib.dump({
            'pipeline': self.pipeline,
            'historical_baselines': self.historical_baselines,
            'metadata': self.metadata
        }, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'MalariaPredictor':
        """Load model from file."""
        data = joblib.load(filepath)
        model = cls(historical_baselines=data['historical_baselines'])
        model.pipeline = data['pipeline']
        model.metadata = data['metadata']
        return model


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_predictions_vs_actuals(df: pd.DataFrame,
                                 predictions: np.ndarray,
                                 save_path: Optional[str] = None):
    """
    Create comprehensive visualization of model predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Province, date, malaria_cases columns.
    predictions : np.ndarray
        Model predictions.
    save_path : str, optional
        Path to save figure.
    """
    df = df.copy()
    df['predicted'] = predictions
    df['error'] = df['malaria_cases'] - df['predicted']
    df['error_pct'] = (df['error'] / df['malaria_cases'] * 100)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    provinces = ['Gaza', 'Inhambane', 'Maputo']
    colors = {'actual': '#2E86AB', 'predicted': '#E94F37'}

    # Time series for each province
    for idx, province in enumerate(provinces):
        ax = axes[idx, 0]
        prov_data = df[df['Province'] == province].sort_values('date')

        ax.plot(prov_data['date'], prov_data['malaria_cases'],
                'o-', color=colors['actual'], label='Actual', linewidth=2, markersize=6)
        ax.plot(prov_data['date'], prov_data['predicted'],
                's--', color=colors['predicted'], label='Predicted', linewidth=2, markersize=6)

        ax.fill_between(prov_data['date'],
                        prov_data['malaria_cases'],
                        prov_data['predicted'],
                        alpha=0.3, color='gray')

        ax.set_title(f'{province} Province - Predictions vs Actuals', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Malaria Cases')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add R² annotation
        prov_r2 = r2_score(prov_data['malaria_cases'], prov_data['predicted'])
        prov_mape = np.mean(np.abs(prov_data['error_pct']))
        ax.text(0.02, 0.98, f'R²={prov_r2:.3f}\nMAPE={prov_mape:.1f}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Scatter plots
    for idx, province in enumerate(provinces):
        ax = axes[idx, 1]
        prov_data = df[df['Province'] == province]

        ax.scatter(prov_data['malaria_cases'], prov_data['predicted'],
                   alpha=0.7, s=80, c=colors['actual'], edgecolors='white')

        # Perfect prediction line
        max_val = max(prov_data['malaria_cases'].max(), prov_data['predicted'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect prediction')

        ax.set_title(f'{province} - Actual vs Predicted Scatter', fontsize=12, fontweight='bold')
        ax.set_xlabel('Actual Cases')
        ax.set_ylabel('Predicted Cases')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def plot_model_summary(train_metrics: Dict,
                       test_metrics: Dict,
                       province_metrics: Dict,
                       save_path: Optional[str] = None):
    """
    Create summary visualization of model performance.

    Parameters
    ----------
    train_metrics : dict
        Training set metrics.
    test_metrics : dict
        Test set metrics.
    province_metrics : dict
        Per-province test metrics.
    save_path : str, optional
        Path to save figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Train vs Test comparison
    ax1 = axes[0]
    metrics = ['R²', 'RMSE', 'MAE']
    train_vals = [train_metrics['r2'], train_metrics['rmse']/1000, train_metrics['mae']/1000]
    test_vals = [test_metrics['r2'], test_metrics['rmse']/1000, test_metrics['mae']/1000]

    x = np.arange(len(metrics))
    width = 0.35

    ax1.bar(x - width/2, train_vals, width, label='Train', color='#2E86AB')
    ax1.bar(x + width/2, test_vals, width, label='Test', color='#E94F37')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['R²', 'RMSE (K)', 'MAE (K)'])
    ax1.set_title('Train vs Test Performance', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: R² by Province
    ax2 = axes[1]
    provinces = list(province_metrics.keys())
    r2_vals = [province_metrics[p]['r2'] for p in provinces]
    colors = ['#E94F37' if r < 0 else '#2E86AB' for r in r2_vals]

    bars = ax2.bar(provinces, r2_vals, color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Test R² by Province', fontweight='bold')
    ax2.set_ylabel('R² Score')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, r2_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # Plot 3: MAPE by Province
    ax3 = axes[2]
    mape_vals = [province_metrics[p]['mape'] for p in provinces]
    colors = ['#E94F37' if m > 50 else '#F5A623' if m > 30 else '#2E86AB' for m in mape_vals]

    bars = ax3.bar(provinces, mape_vals, color=colors)
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Good (<30%)')
    ax3.set_title('Test MAPE by Province', fontweight='bold')
    ax3.set_ylabel('MAPE (%)')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, mape_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()
    return fig


# ==============================================================================
# MAIN TRAINING SCRIPT
# ==============================================================================

def train_and_save_production_model(data_path: str,
                                     model_path: str = 'malaria_model.joblib',
                                     test_year: int = 2024):
    """
    Train production model and save with visualizations.

    Parameters
    ----------
    data_path : str
        Path to training data CSV.
    model_path : str
        Path to save trained model.
    test_year : int
        Year to use for testing.

    Returns
    -------
    tuple
        (model, train_metrics, test_metrics, province_metrics)
    """
    print("="*60)
    print("MALARIA EARLY WARNING SYSTEM")
    print("Production Model Training")
    print("="*60)

    # Load data
    df = load_data(data_path)
    print(f"\nLoaded {len(df)} observations")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Split
    train_df = df[df['Year'] < test_year].copy()
    test_df = df[df['Year'] == test_year].copy()
    print(f"Train: {len(train_df)} rows ({train_df['Year'].min()}-{train_df['Year'].max()})")
    print(f"Test: {len(test_df)} rows (Year {test_year})")

    # Train model
    print("\nTraining regime-aware model...")
    model = MalariaPredictor()
    model.fit(train_df)
    print("Training complete!")

    # Add regime features for evaluation
    train_df = add_regime_features(train_df)
    test_df = add_regime_features(test_df)
    train_df = train_df.dropna(subset=['baseline_ratio_lag1'])
    test_df = test_df.dropna(subset=['baseline_ratio_lag1'])

    # Evaluate
    train_metrics = model.evaluate(train_df)
    test_metrics = model.evaluate(test_df)

    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"\n{'Metric':<15} {'Train':<15} {'Test':<15}")
    print("-"*45)
    print(f"{'R²':<15} {train_metrics['r2']:<15.4f} {test_metrics['r2']:<15.4f}")
    print(f"{'RMSE':<15} {train_metrics['rmse']:<15,.0f} {test_metrics['rmse']:<15,.0f}")
    print(f"{'MAE':<15} {train_metrics['mae']:<15,.0f} {test_metrics['mae']:<15,.0f}")
    print(f"{'MAPE':<15} {train_metrics['mape']:<15.1f}% {test_metrics['mape']:<15.1f}%")

    # Per-province metrics
    province_metrics = {}
    predictions = model.predict(test_df)

    print(f"\n{'='*60}")
    print("PER-PROVINCE TEST PERFORMANCE")
    print(f"{'='*60}")

    for province in PROVINCES:
        mask = test_df['Province'] == province
        prov_true = test_df.loc[mask, 'malaria_cases']
        prov_pred = predictions[mask.values]

        province_metrics[province] = {
            'r2': r2_score(prov_true, prov_pred),
            'rmse': np.sqrt(mean_squared_error(prov_true, prov_pred)),
            'mae': mean_absolute_error(prov_true, prov_pred),
            'mape': np.mean(np.abs((prov_true - prov_pred) / prov_true)) * 100
        }

        pm = province_metrics[province]
        print(f"\n{province}:")
        print(f"  R²: {pm['r2']:.4f}, RMSE: {pm['rmse']:,.0f}, MAPE: {pm['mape']:.1f}%")

    # Save model
    model.save(model_path)

    # Generate visualizations
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    plot_predictions_vs_actuals(test_df, predictions,
                                 save_path='predictions_vs_actuals_2024.png')

    plot_model_summary(train_metrics, test_metrics, province_metrics,
                       save_path='model_performance_summary.png')

    # Save predictions
    results = test_df[['Province', 'Year', 'Month', 'month_english',
                       'Season', 'malaria_cases']].copy()
    results['predicted_cases'] = predictions.astype(int)
    results['error'] = results['malaria_cases'] - results['predicted_cases']
    results['error_pct'] = (results['error'] / results['malaria_cases'] * 100).round(1)
    results.to_csv('final_predictions_2024.csv', index=False)
    print("Saved: final_predictions_2024.csv")

    print(f"\n{'='*60}")
    print("PRODUCTION MODEL READY")
    print(f"{'='*60}")
    print(f"""
    Files created:
    - {model_path} (trained model)
    - predictions_vs_actuals_2024.png (time series plots)
    - model_performance_summary.png (metrics summary)
    - final_predictions_2024.csv (detailed predictions)

    To use the model:

        from malaria_model import MalariaPredictor
        model = MalariaPredictor.load('{model_path}')
        predictions = model.predict(new_data)
    """)

    return model, train_metrics, test_metrics, province_metrics


if __name__ == "__main__":
    model, train_metrics, test_metrics, province_metrics = train_and_save_production_model(
        data_path='master_climate_malaria_provinces_2020_2024.csv',
        model_path='malaria_model.joblib',
        test_year=2024
    )
