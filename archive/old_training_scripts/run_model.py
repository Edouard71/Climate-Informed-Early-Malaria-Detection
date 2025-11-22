"""
Malaria Early Warning System - Run & Visualize
===============================================
Simple script to load the trained model, make predictions,
and generate visualizations.

Usage:
    python run_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from malaria_model import (
    MalariaPredictor,
    load_data,
    add_regime_features,
    plot_predictions_vs_actuals,
    HISTORICAL_BASELINES,
    PROVINCES
)
from sklearn.metrics import r2_score


def run_predictions():
    """Load model and generate predictions on 2024 data."""

    print("="*60)
    print("MALARIA EARLY WARNING SYSTEM")
    print("="*60)

    # 1. Load the trained model
    print("\n1. Loading trained model...")
    model = MalariaPredictor.load('malaria_model.joblib')
    print(f"   Model loaded from: malaria_model.joblib")
    print(f"   Trained on: {model.metadata.get('train_samples', 'N/A')} samples")

    # 2. Load and prepare data
    print("\n2. Loading data...")
    df = load_data('master_climate_malaria_provinces_2020_2024.csv')
    df = add_regime_features(df)
    df = df.dropna(subset=['baseline_ratio_lag1'])

    # Split into train/test
    test_df = df[df['Year'] == 2024].copy()
    print(f"   Test data: {len(test_df)} observations (Year 2024)")

    # 3. Generate predictions
    print("\n3. Generating predictions...")
    predictions = model.predict(test_df)
    test_df['predicted_cases'] = predictions.astype(int)
    test_df['error'] = test_df['malaria_cases'] - test_df['predicted_cases']
    test_df['error_pct'] = (test_df['error'] / test_df['malaria_cases'] * 100).round(1)

    # 4. Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)

    for province in PROVINCES:
        prov_data = test_df[test_df['Province'] == province]
        actual_total = prov_data['malaria_cases'].sum()
        pred_total = prov_data['predicted_cases'].sum()
        prov_r2 = r2_score(prov_data['malaria_cases'], prov_data['predicted_cases'])
        prov_mape = np.mean(np.abs(prov_data['error_pct']))

        print(f"\n{province}:")
        print(f"  Total Actual:    {actual_total:>10,} cases")
        print(f"  Total Predicted: {pred_total:>10,} cases")
        print(f"  R²: {prov_r2:.3f}, MAPE: {prov_mape:.1f}%")
        print(f"\n  Monthly breakdown:")
        print(prov_data[['Month', 'month_english', 'Season', 'malaria_cases',
                        'predicted_cases', 'error_pct']].to_string(index=False))

    return test_df, predictions


def visualize_results(test_df: pd.DataFrame, predictions: np.ndarray):
    """Generate and display visualizations."""

    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    colors = {'actual': '#2E86AB', 'predicted': '#E94F37'}

    # Time series for each province
    for idx, province in enumerate(PROVINCES):
        ax = axes[0, idx]
        prov_data = test_df[test_df['Province'] == province].sort_values('Month')

        months = prov_data['Month']
        actual = prov_data['malaria_cases']
        predicted = prov_data['predicted_cases']

        ax.bar(months - 0.2, actual, 0.4, label='Actual', color=colors['actual'], alpha=0.8)
        ax.bar(months + 0.2, predicted, 0.4, label='Predicted', color=colors['predicted'], alpha=0.8)

        ax.set_xlabel('Month')
        ax.set_ylabel('Malaria Cases')
        ax.set_title(f'{province} - 2024 Monthly Cases', fontsize=12, fontweight='bold')
        ax.legend()
        ax.set_xticks(range(1, 13))
        ax.grid(True, alpha=0.3, axis='y')

        # Add R² annotation
        prov_r2 = r2_score(actual, predicted)
        ax.text(0.02, 0.98, f'R²={prov_r2:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

    # Scatter plots
    for idx, province in enumerate(PROVINCES):
        ax = axes[1, idx]
        prov_data = test_df[test_df['Province'] == province]

        actual = prov_data['malaria_cases']
        predicted = prov_data['predicted_cases']

        # Color by season
        wet_mask = prov_data['Season'] == 'Wet'
        ax.scatter(actual[wet_mask], predicted[wet_mask],
                   s=100, c='blue', alpha=0.7, label='Wet Season', edgecolors='white')
        ax.scatter(actual[~wet_mask], predicted[~wet_mask],
                   s=100, c='orange', alpha=0.7, label='Dry Season', edgecolors='white')

        # Perfect prediction line
        max_val = max(actual.max(), predicted.max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect')

        ax.set_xlabel('Actual Cases')
        ax.set_ylabel('Predicted Cases')
        ax.set_title(f'{province} - Actual vs Predicted', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_results_2024.png', dpi=150, bbox_inches='tight')
    print("\nSaved: model_results_2024.png")

    # Show the plot
    plt.show()

    return fig


def plot_full_timeseries(data_path: str = 'master_climate_malaria_provinces_2020_2024.csv'):
    """Plot full time series with train/test split highlighted."""

    print("\n" + "="*60)
    print("FULL TIME SERIES VISUALIZATION")
    print("="*60)

    # Load all data
    df = load_data(data_path)
    df = add_regime_features(df)
    df = df.dropna(subset=['baseline_ratio_lag1'])

    # Load model and predict on all data
    model = MalariaPredictor.load('malaria_model.joblib')
    predictions = model.predict(df)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for idx, province in enumerate(PROVINCES):
        ax = axes[idx]
        prov_data = df[df['Province'] == province].sort_values('date').copy()
        prov_pred = predictions[df['Province'] == province]

        # Plot actual
        ax.plot(prov_data['date'], prov_data['malaria_cases'],
                'o-', color='#2E86AB', label='Actual', linewidth=2, markersize=4)

        # Plot predicted
        ax.plot(prov_data['date'], prov_pred,
                's--', color='#E94F37', label='Predicted', linewidth=2, markersize=4, alpha=0.8)

        # Shade 2024 (test period)
        test_start = pd.Timestamp('2024-01-01')
        test_end = pd.Timestamp('2024-12-31')
        ax.axvspan(test_start, test_end, alpha=0.2, color='yellow', label='Test Period (2024)')

        ax.set_ylabel('Malaria Cases')
        ax.set_title(f'{province} Province - Full Time Series (2020-2024)',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add historical baseline
        baseline = HISTORICAL_BASELINES[province]['mean']
        ax.axhline(y=baseline, color='green', linestyle=':', alpha=0.5,
                   label=f'Historical baseline ({baseline:,})')

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig('full_timeseries_2020_2024.png', dpi=150, bbox_inches='tight')
    print("\nSaved: full_timeseries_2020_2024.png")

    plt.show()

    return fig


def main():
    """Main function to run model and generate visualizations."""

    # Run predictions
    test_df, predictions = run_predictions()

    # Generate visualizations
    print("\n" + "-"*60)
    input("Press Enter to generate visualizations...")

    visualize_results(test_df, predictions)

    print("\n" + "-"*60)
    input("Press Enter to see full time series...")

    plot_full_timeseries()

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("""
    Generated files:
    - model_results_2024.png (2024 predictions bar chart + scatter)
    - full_timeseries_2020_2024.png (complete time series)

    To make new predictions:

        from malaria_model import MalariaPredictor, add_regime_features

        # Load model
        model = MalariaPredictor.load('malaria_model.joblib')

        # Prepare your data (needs these columns):
        # - iosd_lag1, iosd_lag2, precip_lag1
        # - Season ('Wet' or 'Dry')
        # - Province ('Gaza', 'Inhambane', or 'Maputo')
        # - malaria_cases (for regime features)

        # Add regime features
        data = add_regime_features(your_data)

        # Predict
        predictions = model.predict(data)
    """)


if __name__ == "__main__":
    main()
