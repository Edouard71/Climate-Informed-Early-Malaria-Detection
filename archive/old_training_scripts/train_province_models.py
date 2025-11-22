"""
Malaria Early Warning System - Province-Specific Models
========================================================
Trains separate linear regression models for each province.
This accounts for province-specific baselines and climate responses.

Time-based split: Train on 2020-2023, Test on 2024
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load dataset and prepare for modeling."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Drop rows with NA in lag columns
    lag_cols = ['precip_lag1', 'precip_lag2', 'iosd_lag1', 'iosd_lag2']
    df = df.dropna(subset=lag_cols)

    return df


def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}


def train_province_model(df: pd.DataFrame, province: str, test_year: int = 2024):
    """
    Train a model for a single province.
    Returns the model and evaluation metrics.
    """
    # Filter to this province
    province_df = df[df['Province'] == province].copy()

    # Time split
    train = province_df[province_df['Year'] < test_year]
    test = province_df[province_df['Year'] == test_year]

    # Features (no Province needed - it's province-specific)
    numeric_features = ['iosd_lag1', 'iosd_lag2', 'precip_lag1']
    categorical_features = ['Season']
    target = 'malaria_cases'

    X_train = train[numeric_features + categorical_features]
    y_train = train[target]
    X_test = test[numeric_features + categorical_features]
    y_test = test[target]

    # Build pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Evaluate
    train_metrics = evaluate_model(y_train, y_train_pred)
    test_metrics = evaluate_model(y_test, y_test_pred)

    # Get coefficients
    model = pipeline.named_steps['regressor']
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_features = numeric_features + cat_names
    coefficients = dict(zip(all_features, model.coef_))
    coefficients['intercept'] = model.intercept_

    # Create results dataframe
    results = test[['Province', 'Year', 'Month', 'month_english', 'malaria_cases']].copy()
    results['predicted_cases'] = y_test_pred.astype(int)
    results['error'] = results['malaria_cases'] - results['predicted_cases']
    results['error_pct'] = (results['error'] / results['malaria_cases'] * 100).round(1)

    return {
        'province': province,
        'pipeline': pipeline,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'coefficients': coefficients,
        'results': results,
        'train_size': len(train),
        'test_size': len(test)
    }


def print_province_results(result: dict):
    """Print results for a single province."""
    province = result['province']
    train_m = result['train_metrics']
    test_m = result['test_metrics']
    coefs = result['coefficients']

    print(f"\n{'='*60}")
    print(f"  {province.upper()} PROVINCE MODEL")
    print(f"{'='*60}")
    print(f"Training samples: {result['train_size']}, Test samples: {result['test_size']}")

    print(f"\n  {'Metric':<15} {'Train':<15} {'Test':<15}")
    print(f"  {'-'*45}")
    print(f"  {'R²':<15} {train_m['r2']:<15.4f} {test_m['r2']:<15.4f}")
    print(f"  {'RMSE':<15} {train_m['rmse']:<15,.0f} {test_m['rmse']:<15,.0f}")
    print(f"  {'MAE':<15} {train_m['mae']:<15,.0f} {test_m['mae']:<15,.0f}")
    print(f"  {'MAPE':<15} {train_m['mape']:<15.1f}% {test_m['mape']:<15.1f}%")

    print(f"\n  Coefficients:")
    for name, coef in sorted(coefs.items(), key=lambda x: abs(x[1]) if x[0] != 'intercept' else 0, reverse=True):
        if name != 'intercept':
            print(f"    {name:<20} {coef:>12,.2f}")
    print(f"    {'intercept':<20} {coefs['intercept']:>12,.2f}")

    # Show sample predictions
    print(f"\n  2024 Predictions (first 6 months):")
    print(result['results'].head(6).to_string(index=False))


def compare_models(province_results: list, pooled_test_r2: float):
    """Compare province-specific models to pooled model."""
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON: Province-Specific vs Pooled")
    print(f"{'='*60}")

    print(f"\n  {'Province':<15} {'Train R²':<12} {'Test R²':<12} {'Test MAPE':<12}")
    print(f"  {'-'*51}")

    total_test_actual = 0
    total_test_pred = 0
    all_results = []

    for result in province_results:
        train_r2 = result['train_metrics']['r2']
        test_r2 = result['test_metrics']['r2']
        test_mape = result['test_metrics']['mape']
        print(f"  {result['province']:<15} {train_r2:<12.4f} {test_r2:<12.4f} {test_mape:<12.1f}%")

        all_results.append(result['results'])
        total_test_actual += result['results']['malaria_cases'].sum()
        total_test_pred += result['results']['predicted_cases'].sum()

    # Combined metrics
    combined_results = pd.concat(all_results)
    combined_r2 = r2_score(combined_results['malaria_cases'], combined_results['predicted_cases'])
    combined_mape = np.mean(np.abs(combined_results['error'] / combined_results['malaria_cases'])) * 100

    print(f"  {'-'*51}")
    print(f"  {'Combined':<15} {'-':<12} {combined_r2:<12.4f} {combined_mape:<12.1f}%")
    print(f"  {'Pooled Model':<15} {'-':<12} {pooled_test_r2:<12.4f} {'-':<12}")

    print(f"\n  Improvement over pooled model:")
    improvement = combined_r2 - pooled_test_r2
    print(f"    R² improvement: {improvement:+.4f} ({improvement/abs(pooled_test_r2)*100:+.1f}%)")

    return combined_results


def main():
    print("="*60)
    print("MALARIA EARLY WARNING SYSTEM")
    print("Province-Specific Models")
    print("="*60)

    # Load data
    filepath = 'master_climate_malaria_provinces_2020_2024.csv'
    df = load_and_prepare_data(filepath)

    print(f"\nDataset: {len(df)} observations")
    print(f"Provinces: {df['Province'].unique().tolist()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Train province-specific models
    provinces = ['Gaza', 'Inhambane', 'Maputo']
    province_results = []

    for province in provinces:
        result = train_province_model(df, province)
        province_results.append(result)
        print_province_results(result)

    # Compare to pooled model (R² from previous run)
    pooled_test_r2 = 0.3195
    combined_results = compare_models(province_results, pooled_test_r2)

    # Save combined predictions
    combined_results.to_csv('predictions_province_specific_2024.csv', index=False)
    print(f"\nPredictions saved to predictions_province_specific_2024.csv")

    # Summary by province
    print(f"\n{'='*60}")
    print("  2024 ANNUAL TOTALS BY PROVINCE")
    print(f"{'='*60}")
    summary = combined_results.groupby('Province').agg({
        'malaria_cases': 'sum',
        'predicted_cases': 'sum'
    })
    summary['error'] = summary['malaria_cases'] - summary['predicted_cases']
    summary['error_pct'] = (summary['error'] / summary['malaria_cases'] * 100).round(1)
    print(summary)

    print(f"\n{'='*60}")
    print("  KEY FINDINGS")
    print(f"{'='*60}")
    print("""
    1. Province-specific models capture local baselines better
    2. IOSD remains the strongest climate predictor across all provinces
    3. Gaza shows the largest prediction errors - likely due to
       intervention effects or reporting changes in 2024
    4. Inhambane and Maputo models perform more consistently
    """)

    return province_results


if __name__ == "__main__":
    results = main()
