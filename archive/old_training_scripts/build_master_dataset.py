"""
Master Dataset Builder for Climate-Malaria Early Warning System
================================================================
Southern Mozambique (Maputo, Gaza, Inhambane Provinces)

This script demonstrates how to construct the master dataset from:
1. Malaria case data (Excel/CSV from DHIS2 or Goodbye Malaria)
2. CHIRPS precipitation rasters
3. IOSD and ENSO climate indices

Author: [Your Name]
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional: for spatial data processing
# import rasterio
# import geopandas as gpd
# from rasterstats import zonal_stats


# =============================================================================
# CONFIGURATION
# =============================================================================

PROVINCES = ['Maputo', 'Gaza', 'Inhambane']

POPULATION_ESTIMATES = {
    'Maputo': 2500000,
    'Gaza': 1450000,
    'Inhambane': 1560000
}

START_DATE = '2020-01-01'
END_DATE = '2025-08-01'

# Paths (adjust to your setup)
PATH_MALARIA_DATA = './data/malaria_cases/'
PATH_CHIRPS_DATA = './data/chirps/'
PATH_CLIMATE_INDICES = './data/climate_indices/'
PATH_OUTPUT = './output/master_dataset.csv'

# =============================================================================
# STEP 1: LOAD AND PREPARE MALARIA DATA
# =============================================================================

def load_malaria_data(province):
    """
    Load malaria case data for a given province.
    
    Expected format (similar to data_47.xls):
    - organisationunitname, Month, Year, Malaria - Total casos de Malaria
    
    Returns:
    - DataFrame with columns: province, date, malaria_cases, malaria_incidence
    """
    
    # Example: load from Excel file
    # df = pd.read_excel(f'{PATH_MALARIA_DATA}/{province}_cases.xls')
    
    # For demonstration, we'll create a dummy structure
    # Replace this with your actual data loading
    
    print(f"Loading malaria data for {province}...")
    
    # Dummy data structure - replace with actual loading
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    df = pd.DataFrame({
        'date': dates,
        'malaria_cases': np.nan  # Fill with actual data
    })
    
    df['province'] = province
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Calculate incidence per 1,000 population
    df['malaria_incidence'] = (df['malaria_cases'] / POPULATION_ESTIMATES[province]) * 1000
    
    # Add population
    df['population'] = POPULATION_ESTIMATES[province]
    
    return df[['province', 'year', 'month', 'date', 'malaria_cases', 'malaria_incidence', 'population']]


def consolidate_malaria_data(provinces):
    """Combine malaria data from all provinces."""
    
    all_data = []
    for province in provinces:
        df_province = load_malaria_data(province)
        all_data.append(df_province)
    
    df_malaria = pd.concat(all_data, ignore_index=True)
    df_malaria = df_malaria.sort_values(['province', 'date']).reset_index(drop=True)
    
    print(f"Consolidated malaria data: {len(df_malaria)} rows, {len(provinces)} provinces")
    
    return df_malaria


# =============================================================================
# STEP 2: PROCESS PRECIPITATION DATA (CHIRPS)
# =============================================================================

def extract_chirps_precipitation(province, date):
    """
    Extract mean precipitation for a province from CHIRPS raster.
    
    Process:
    1. Load CHIRPS monthly raster for given date
    2. Crop to province boundary
    3. Mask to land areas
    4. Calculate spatial mean
    
    Returns:
    - Mean precipitation in mm
    """
    
    # Pseudo-code for spatial processing
    # Replace with actual implementation using rasterio/geopandas
    
    # chirps_file = f'{PATH_CHIRPS_DATA}/chirps-v2.0.{date.year}.{date.month:02d}.tif'
    # province_boundary = gpd.read_file(f'./boundaries/{province}.shp')
    # 
    # with rasterio.open(chirps_file) as src:
    #     stats = zonal_stats(province_boundary, src.read(1), 
    #                         affine=src.transform, stats=['mean'])
    #     precip_mean = stats[0]['mean']
    # 
    # return precip_mean
    
    # Dummy return for demonstration
    return np.nan


def build_precipitation_dataframe(provinces):
    """Build precipitation dataset for all provinces and dates."""
    
    print("Processing CHIRPS precipitation data...")
    
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    
    data = []
    for province in provinces:
        for date in dates:
            precip = extract_chirps_precipitation(province, date)
            data.append({
                'province': province,
                'date': date,
                'precip_mm': precip
            })
    
    df_precip = pd.DataFrame(data)
    
    print(f"Precipitation data: {len(df_precip)} rows")
    
    return df_precip


# =============================================================================
# STEP 3: LOAD CLIMATE INDICES (IOSD, ENSO)
# =============================================================================

def load_iosd_index():
    """
    Load IOSD index time series.
    
    Expected format:
    - date, iosd_index (in °C)
    
    Can be calculated from ERA5 SST data or loaded from pre-computed file.
    """
    
    print("Loading IOSD index...")
    
    # Example: load from CSV
    # df_iosd = pd.read_csv(f'{PATH_CLIMATE_INDICES}/iosd_monthly.csv')
    # df_iosd['date'] = pd.to_datetime(df_iosd['date'])
    
    # Dummy data for demonstration
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    df_iosd = pd.DataFrame({
        'date': dates,
        'iosd_index': np.nan  # Fill with actual IOSD values
    })
    
    # Classify IOSD phase (±0.4°C threshold)
    df_iosd['iosd_phase'] = df_iosd['iosd_index'].apply(
        lambda x: 'positive' if x > 0.4 else ('negative' if x < -0.4 else 'neutral') if pd.notna(x) else ''
    )
    
    return df_iosd


def load_enso_oni():
    """
    Load ENSO ONI (Oceanic Niño Index) time series.
    
    Source: NOAA Climate Prediction Center
    URL: https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
    
    Expected format:
    - date, oni (dimensionless, °C anomaly)
    """
    
    print("Loading ENSO ONI...")
    
    # Example: load from CSV downloaded from NOAA
    # df_oni = pd.read_csv(f'{PATH_CLIMATE_INDICES}/oni_monthly.csv')
    # df_oni['date'] = pd.to_datetime(df_oni['date'])
    
    # Dummy data for demonstration
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    df_oni = pd.DataFrame({
        'date': dates,
        'enso_oni': np.nan  # Fill with actual ONI values
    })
    
    # Classify ENSO phase (±0.5°C threshold)
    df_oni['enso_phase'] = df_oni['enso_oni'].apply(
        lambda x: 'el_nino' if x >= 0.5 else ('la_nina' if x <= -0.5 else 'neutral') if pd.notna(x) else ''
    )
    
    return df_oni


def build_climate_indices_dataframe(provinces):
    """
    Build climate indices dataset.
    
    Note: IOSD and ENSO are regional indices (same value for all provinces)
    so we replicate them for each province to maintain long format.
    """
    
    df_iosd = load_iosd_index()
    df_oni = load_enso_oni()
    
    # Merge indices
    df_indices = df_iosd.merge(df_oni, on='date', how='outer')
    
    # Replicate for each province (long format)
    data = []
    for province in provinces:
        df_temp = df_indices.copy()
        df_temp['province'] = province
        data.append(df_temp)
    
    df_climate_indices = pd.concat(data, ignore_index=True)
    
    print(f"Climate indices data: {len(df_climate_indices)} rows")
    
    return df_climate_indices


# =============================================================================
# STEP 4: CREATE LAGGED VARIABLES
# =============================================================================

def create_lagged_variables(df, var_name, max_lag=3):
    """
    Create lagged versions of a variable, grouped by province.
    
    Args:
    - df: DataFrame with province and date columns
    - var_name: Name of variable to lag (e.g., 'precip_mm')
    - max_lag: Maximum number of lags to create
    
    Returns:
    - DataFrame with additional lagged columns
    """
    
    df = df.sort_values(['province', 'date']).copy()
    
    for lag in range(1, max_lag + 1):
        lag_col = f'{var_name}_lag{lag}'
        df[lag_col] = df.groupby('province')[var_name].shift(lag)
    
    return df


# =============================================================================
# STEP 5: ADD DERIVED VARIABLES
# =============================================================================

def add_derived_variables(df):
    """Add season indicators and other derived variables."""
    
    # Wet season: November-April (months 11, 12, 1, 2, 3, 4)
    # Dry season: May-October (months 5, 6, 7, 8, 9, 10)
    
    df['season'] = df['month'].apply(
        lambda m: 'wet' if m in [11, 12, 1, 2, 3, 4] else 'dry'
    )
    
    df['wet_season'] = (df['season'] == 'wet').astype(int)
    
    # Month name (optional, for readability)
    df['month_name'] = df['date'].dt.month_name()
    
    return df


# =============================================================================
# STEP 6: MERGE AND FINALIZE MASTER DATASET
# =============================================================================

def build_master_dataset(provinces):
    """
    Main function to build complete master dataset.
    
    Process:
    1. Load malaria data
    2. Load precipitation data
    3. Load climate indices
    4. Merge everything
    5. Create lagged variables
    6. Add derived variables
    7. Final cleanup and validation
    """
    
    print("=" * 70)
    print("BUILDING MASTER DATASET FOR CLIMATE-MALARIA EARLY WARNING")
    print("=" * 70)
    
    # Step 1: Malaria data
    df_malaria = consolidate_malaria_data(provinces)
    
    # Step 2: Precipitation data
    df_precip = build_precipitation_dataframe(provinces)
    
    # Step 3: Climate indices
    df_indices = build_climate_indices_dataframe(provinces)
    
    # Step 4: Merge all datasets
    print("\nMerging datasets...")
    master = df_malaria.merge(df_precip, on=['province', 'date'], how='left')
    master = master.merge(df_indices, on=['province', 'date'], how='left')
    
    # Step 5: Create lagged variables
    print("Creating lagged variables...")
    master = create_lagged_variables(master, 'precip_mm', max_lag=3)
    master = create_lagged_variables(master, 'iosd_index', max_lag=3)
    master = create_lagged_variables(master, 'enso_oni', max_lag=3)
    
    # Step 6: Add derived variables
    print("Adding derived variables...")
    master = add_derived_variables(master)
    
    # Step 7: Final column ordering
    column_order = [
        'province', 'year', 'month', 'date',
        'malaria_cases', 'malaria_incidence',
        'precip_mm', 'precip_lag1', 'precip_lag2', 'precip_lag3',
        'iosd_index', 'iosd_lag1', 'iosd_lag2', 'iosd_lag3', 'iosd_phase',
        'enso_oni', 'enso_lag1', 'enso_lag2', 'enso_lag3', 'enso_phase',
        'season', 'wet_season', 'month_name',
        'population'
    ]
    
    # Keep only columns that exist
    column_order = [col for col in column_order if col in master.columns]
    master = master[column_order]
    
    # Step 8: Validation
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total rows: {len(master)}")
    print(f"Date range: {master['date'].min()} to {master['date'].max()}")
    print(f"Provinces: {master['province'].unique().tolist()}")
    print(f"\nColumns: {list(master.columns)}")
    print(f"\nMissing values:")
    print(master.isnull().sum())
    print(f"\nFirst 5 rows:")
    print(master.head())
    
    return master


# =============================================================================
# STEP 7: SAVE MASTER DATASET
# =============================================================================

def save_master_dataset(df, output_path=PATH_OUTPUT):
    """Save master dataset to CSV."""
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Master dataset saved to: {output_path}")
    
    # Also save a summary report
    summary_path = output_path.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("MASTER DATASET SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Date range: {df['date'].min()} to {df['date'].max()}\n")
        f.write(f"Provinces: {', '.join(df['province'].unique())}\n")
        f.write(f"\nColumns ({len(df.columns)}):\n")
        for col in df.columns:
            f.write(f"  - {col}\n")
        f.write(f"\nMissing values:\n")
        f.write(str(df.isnull().sum()))
    
    print(f"✓ Summary saved to: {summary_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    
    # Build the master dataset
    master_df = build_master_dataset(PROVINCES)
    
    # Save to file
    save_master_dataset(master_df, PATH_OUTPUT)
    
    print("\n" + "=" * 70)
    print("MASTER DATASET BUILD COMPLETE")
    print("=" * 70)
    
    # Optional: Quick quality checks
    print("\n📊 Quick quality checks:")
    
    # Check date continuity
    for province in PROVINCES:
        df_prov = master_df[master_df['province'] == province]
        expected_months = len(pd.date_range(START_DATE, END_DATE, freq='MS'))
        actual_months = len(df_prov)
        if expected_months == actual_months:
            print(f"  ✓ {province}: {actual_months} months (complete)")
        else:
            print(f"  ⚠ {province}: {actual_months} months (expected {expected_months})")
    
    # Check for duplicate dates per province
    duplicates = master_df.duplicated(subset=['province', 'date']).sum()
    if duplicates == 0:
        print(f"  ✓ No duplicate province-date combinations")
    else:
        print(f"  ⚠ Found {duplicates} duplicate province-date combinations")
    
    # Check lag variable consistency
    for var in ['precip', 'iosd_index', 'enso_oni']:
        lag1_col = f'{var}_lag1'
        if lag1_col in master_df.columns:
            # For each province, check if lag1 matches previous month's value
            for province in PROVINCES:
                df_prov = master_df[master_df['province'] == province].sort_values('date').reset_index(drop=True)
                if len(df_prov) > 1:
                    # Compare lag1 with shifted original
                    expected_lag1 = df_prov[var].shift(1)
                    if df_prov[lag1_col].equals(expected_lag1):
                        print(f"  ✓ {province} {lag1_col}: correctly calculated")
    
    print("\n✅ Done! Your master dataset is ready for modeling.")
    print(f"\nNext steps:")
    print("  1. Fill in actual malaria case data")
    print("  2. Process CHIRPS rasters to extract precipitation")
    print("  3. Calculate or load IOSD and ENSO indices")
    print("  4. Run this script to generate complete dataset")
    print("  5. Proceed with statistical modeling")
