"""
Malaria Early Warning System - Linear Regression Model
=======================================================
Predicts malaria cases using climate variables (IOSD, precipitation).
Based on research showing IOSD as strongest predictor (R²=0.40-0.45).

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

    # Convert date column
    df['date'] = pd.to_datetime(df['date'])

    # Drop rows with NA in lag columns (first 2 months per province)
    lag_cols = ['precip_lag1', 'precip_lag2', 'iosd_lag1', 'iosd_lag2',
                'enso_lag1', 'enso_lag2']
    df = df.dropna(subset=lag_cols)

    print(f"Dataset shape after dropping NA: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Provinces: {df['Province'].unique()}")

    return df


def create_time_split(df: pd.DataFrame, test_year: int = 2024):
    """
    Time-based train/test split.
    Train: All years before test_year
    Test: test_year only
    """
    train = df[df['Year'] < test_year].copy()
    test = df[df['Year'] == test_year].copy()

    print(f"\nTrain set: {len(train)} rows ({train['Year'].min()}-{train['Year'].max()})")
    print(f"Test set: {len(test)} rows (Year {test_year})")

    return train, test


def select_features(df: pd.DataFrame):
    """
    Select features based on research findings.
    Primary: IOSD lag1/lag2 (strongest predictor)
    Secondary: Precipitation lag1
    Control: Season, Province

    Note: ENSO excluded due to weak signal in this region.
    """
    # Feature columns
    numeric_features = [
        'iosd_lag1',      # Primary predictor (R²=0.45)
        'iosd_lag2',      # Secondary IOSD signal
        'precip_lag1',    # Precipitation with 1-month lag
    ]

    categorical_features = [
        'Season',         # Wet/Dry season
        'Province',       # Province-level variation
    ]

    target = 'malaria_cases'

    return numeric_features, categorical_features, target


def build_pipeline(numeric_features: list, categorical_features: list):
    """Build sklearn pipeline with preprocessing and model."""

    # Preprocessor: scale numeric, one-hot encode categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    # Full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    return pipeline


def evaluate_model(y_true, y_pred, dataset_name: str = ""):
    """Calculate and print evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{'='*50}")
    print(f"{dataset_name} Evaluation Metrics")
    print(f"{'='*50}")
    print(f"R² Score:           {r2:.4f}")
    print(f"RMSE:               {rmse:,.0f} cases")
    print(f"MAE:                {mae:,.0f} cases")
    print(f"MAPE:               {mape:.1f}%")

    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}


def analyze_coefficients(pipeline, numeric_features: list, categorical_features: list):
    """Extract and display model coefficients."""
    model = pipeline.named_steps['regressor']
    preprocessor = pipeline.named_steps['preprocessor']

    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()

    all_features = numeric_features + cat_feature_names
    coefficients = model.coef_

    print(f"\n{'='*50}")
    print("Model Coefficients")
    print(f"{'='*50}")
    print(f"{'Feature':<25} {'Coefficient':>15}")
    print("-" * 42)

    for name, coef in sorted(zip(all_features, coefficients), key=lambda x: abs(x[1]), reverse=True):
        print(f"{name:<25} {coef:>15,.2f}")

    print(f"\nIntercept: {model.intercept_:,.2f}")


def generate_predictions_report(test_df: pd.DataFrame, y_pred: np.ndarray):
    """Generate detailed predictions vs actuals report."""
    results = test_df[['Province', 'Year', 'Month', 'month_english', 'malaria_cases']].copy()
    results['predicted_cases'] = y_pred.astype(int)
    results['error'] = results['malaria_cases'] - results['predicted_cases']
    results['error_pct'] = (results['error'] / results['malaria_cases'] * 100).round(1)

    print(f"\n{'='*50}")
    print("2024 Predictions vs Actuals (Sample)")
    print(f"{'='*50}")
    print(results.head(15).to_string(index=False))

    # Summary by province
    print(f"\n{'='*50}")
    print("Summary by Province (2024)")
    print(f"{'='*50}")
    summary = results.groupby('Province').agg({
        'malaria_cases': 'sum',
        'predicted_cases': 'sum',
        'error': 'sum'
    }).round(0)
    summary['error_pct'] = (summary['error'] / summary['malaria_cases'] * 100).round(1)
    print(summary)

    return results


def main():
    print("="*60)
    print("MALARIA EARLY WARNING SYSTEM - Model Training")
    print("="*60)

    # Load data
    filepath = 'master_climate_malaria_provinces_2020_2024.csv'
    df = load_and_prepare_data(filepath)

    # Time-based split
    train_df, test_df = create_time_split(df, test_year=2024)

    # Select features
    numeric_features, categorical_features, target = select_features(df)

    print(f"\nFeatures used:")
    print(f"  Numeric: {numeric_features}")
    print(f"  Categorical: {categorical_features}")
    print(f"  Target: {target}")

    # Prepare X and y
    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df[target]
    X_test = test_df[numeric_features + categorical_features]
    y_test = test_df[target]

    # Build and train pipeline
    print("\nTraining Linear Regression model...")
    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)
    print("Training complete!")

    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Evaluate
    train_metrics = evaluate_model(y_train, y_train_pred, "Training Set (2020-2023)")
    test_metrics = evaluate_model(y_test, y_test_pred, "Test Set (2024)")

    # Analyze coefficients
    analyze_coefficients(pipeline, numeric_features, categorical_features)

    # Detailed predictions report
    results = generate_predictions_report(test_df, y_test_pred)

    # Save results
    results.to_csv('predictions_2024.csv', index=False)
    print(f"\nPredictions saved to predictions_2024.csv")

    print(f"\n{'='*60}")
    print("MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"Model: Linear Regression")
    print(f"Training R²: {train_metrics['r2']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"Test MAPE: {test_metrics['mape']:.1f}%")
    print(f"\nKey insight: IOSD lag1/lag2 are the primary predictors,")
    print(f"matching research findings (R²=0.40-0.45 for IOSD alone).")

    return pipeline, results


if __name__ == "__main__":
    pipeline, results = main()
