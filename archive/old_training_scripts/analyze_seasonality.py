"""
Analyze Gaza Seasonality and Build Season-Aware Model
======================================================
Gaza shows extreme dry season drops - we need to model this better.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    lag_cols = ['precip_lag1', 'precip_lag2', 'iosd_lag1', 'iosd_lag2']
    df = df.dropna(subset=lag_cols)
    return df


def plot_gaza_seasonality(df: pd.DataFrame):
    """Plot Gaza cases by year to see seasonal pattern."""
    gaza = df[df['Province'] == 'Gaza'].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Monthly cases by year
    ax1 = axes[0, 0]
    for year in gaza['Year'].unique():
        year_data = gaza[gaza['Year'] == year]
        ax1.plot(year_data['Month'], year_data['malaria_cases'],
                marker='o', label=str(year), linewidth=2)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Malaria Cases')
    ax1.set_title('Gaza: Monthly Cases by Year')
    ax1.legend()
    ax1.set_xticks(range(1, 13))
    ax1.grid(True, alpha=0.3)

    # Plot 2: Wet vs Dry season comparison
    ax2 = axes[0, 1]
    seasonal_avg = gaza.groupby(['Year', 'Season'])['malaria_cases'].mean().unstack()
    seasonal_avg.plot(kind='bar', ax=ax2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Average Monthly Cases')
    ax2.set_title('Gaza: Wet vs Dry Season Average Cases')
    ax2.legend(title='Season')
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Time series with season shading
    ax3 = axes[1, 0]
    ax3.plot(gaza['date'], gaza['malaria_cases'], 'b-', linewidth=1.5)
    # Shade wet seasons
    for year in gaza['Year'].unique():
        # Wet season: Nov-Apr (spanning years)
        wet_start = pd.Timestamp(f'{year}-11-01')
        wet_end = pd.Timestamp(f'{year+1}-04-30')
        ax3.axvspan(wet_start, wet_end, alpha=0.2, color='blue', label='Wet' if year == gaza['Year'].min() else '')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Malaria Cases')
    ax3.set_title('Gaza: Cases Over Time (blue shading = wet season)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cases vs Precipitation colored by season
    ax4 = axes[1, 1]
    wet = gaza[gaza['Season'] == 'Wet']
    dry = gaza[gaza['Season'] == 'Dry']
    ax4.scatter(wet['precip_lag1'], wet['malaria_cases'], c='blue', alpha=0.6, label='Wet', s=60)
    ax4.scatter(dry['precip_lag1'], dry['malaria_cases'], c='orange', alpha=0.6, label='Dry', s=60)
    ax4.set_xlabel('Precipitation Lag-1 (mm)')
    ax4.set_ylabel('Malaria Cases')
    ax4.set_title('Gaza: Cases vs Precipitation by Season')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gaza_seasonality_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: gaza_seasonality_analysis.png")

    # Print seasonal statistics
    print("\n" + "="*60)
    print("GAZA SEASONAL STATISTICS")
    print("="*60)

    seasonal_stats = gaza.groupby('Season')['malaria_cases'].agg(['mean', 'std', 'min', 'max'])
    print(f"\n{seasonal_stats}")

    wet_mean = gaza[gaza['Season'] == 'Wet']['malaria_cases'].mean()
    dry_mean = gaza[gaza['Season'] == 'Dry']['malaria_cases'].mean()
    ratio = wet_mean / dry_mean
    print(f"\nWet/Dry ratio: {ratio:.2f}x")
    print(f"Dry season cases are {(1-1/ratio)*100:.0f}% lower than wet season")


def build_season_interaction_model(df: pd.DataFrame, province: str = 'Gaza'):
    """
    Build model with season interactions.
    This allows climate effects to differ between wet and dry seasons.
    """
    province_df = df[df['Province'] == province].copy()

    # Create interaction features
    province_df['is_wet'] = (province_df['Season'] == 'Wet').astype(int)
    province_df['iosd_lag1_wet'] = province_df['iosd_lag1'] * province_df['is_wet']
    province_df['iosd_lag1_dry'] = province_df['iosd_lag1'] * (1 - province_df['is_wet'])
    province_df['precip_lag1_wet'] = province_df['precip_lag1'] * province_df['is_wet']
    province_df['precip_lag1_dry'] = province_df['precip_lag1'] * (1 - province_df['is_wet'])

    # Time split
    train = province_df[province_df['Year'] < 2024]
    test = province_df[province_df['Year'] == 2024]

    # Features with interactions
    features = [
        'is_wet',           # Season baseline
        'iosd_lag1_wet',    # IOSD effect in wet season
        'iosd_lag1_dry',    # IOSD effect in dry season
        'precip_lag1_wet',  # Precip effect in wet season
        'precip_lag1_dry',  # Precip effect in dry season
    ]

    X_train = train[features]
    y_train = train['malaria_cases']
    X_test = test[features]
    y_test = test['malaria_cases']

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    print(f"\n{'='*60}")
    print(f"{province.upper()} - SEASON INTERACTION MODEL")
    print(f"{'='*60}")
    print(f"\nFeatures: {features}")
    print(f"\nCoefficients:")
    for name, coef in zip(features, model.coef_):
        print(f"  {name:<20}: {coef:>12,.2f}")
    print(f"  {'intercept':<20}: {model.intercept_:>12,.2f}")

    print(f"\nPerformance:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:,.0f} cases")
    print(f"  Test MAPE: {test_mape:.1f}%")

    # Show predictions
    results = test[['Month', 'month_english', 'Season', 'malaria_cases']].copy()
    results['predicted'] = y_test_pred.astype(int)
    results['error'] = results['malaria_cases'] - results['predicted']
    results['error_pct'] = (results['error'] / results['malaria_cases'] * 100).round(1)

    print(f"\n2024 Predictions:")
    print(results.to_string(index=False))

    return model, scaler, results, {'train_r2': train_r2, 'test_r2': test_r2, 'test_mape': test_mape}


def build_all_province_models(df: pd.DataFrame):
    """Build season-interaction models for all provinces."""
    all_results = []
    all_metrics = {}

    for province in ['Gaza', 'Inhambane', 'Maputo']:
        model, scaler, results, metrics = build_season_interaction_model(df, province)
        results['Province'] = province
        all_results.append(results)
        all_metrics[province] = metrics

    # Combined results
    combined = pd.concat(all_results)
    combined_r2 = r2_score(combined['malaria_cases'], combined['predicted'])
    combined_mape = np.mean(np.abs((combined['malaria_cases'] - combined['predicted']) / combined['malaria_cases'])) * 100

    print(f"\n{'='*60}")
    print("COMBINED RESULTS - SEASON INTERACTION MODELS")
    print(f"{'='*60}")
    print(f"\n{'Province':<15} {'Train R²':<12} {'Test R²':<12} {'Test MAPE':<12}")
    print("-" * 51)
    for province, metrics in all_metrics.items():
        print(f"{province:<15} {metrics['train_r2']:<12.4f} {metrics['test_r2']:<12.4f} {metrics['test_mape']:<12.1f}%")
    print("-" * 51)
    print(f"{'Combined':<15} {'-':<12} {combined_r2:<12.4f} {combined_mape:<12.1f}%")

    print(f"\n\nComparison to previous models:")
    print(f"  Pooled model (no interactions):        Test R² = 0.32")
    print(f"  Province-specific (no interactions):   Test R² = 0.30")
    print(f"  Season-interaction models:             Test R² = {combined_r2:.2f}")

    combined.to_csv('predictions_season_interaction_2024.csv', index=False)
    print(f"\nSaved: predictions_season_interaction_2024.csv")

    return combined


def main():
    print("="*60)
    print("SEASONALITY ANALYSIS & IMPROVED MODELS")
    print("="*60)

    df = load_data('master_climate_malaria_provinces_2020_2024.csv')

    # Analyze Gaza seasonality
    plot_gaza_seasonality(df)

    # Build improved models with season interactions
    results = build_all_province_models(df)

    print(f"\n{'='*60}")
    print("KEY INSIGHT")
    print(f"{'='*60}")
    print("""
    The dry season in Gaza shows dramatically lower cases because:
    1. Less precipitation → fewer mosquito breeding sites
    2. Lower humidity → shorter mosquito lifespan
    3. The IOSD/precip effects operate differently in each season

    By modeling wet/dry season separately, we capture this dynamic
    better than a simple Season dummy variable.
    """)


if __name__ == "__main__":
    main()
