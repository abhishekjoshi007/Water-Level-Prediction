
import pandas as pd
import numpy as np

print("="*80)
print("DIAGNOSTIC: Identifying Missing Features in 2025 Test Data")
print("="*80)

# Load 2025 data
water_2025 = pd.read_excel('/Users/abhishekjoshi/Documents/GitHub/Water-Level-Prediction/Data/test_data/Coastal Bend Water Level Measurements_2025.xlsx')
met_2025 = pd.read_excel('/Users/abhishekjoshi/Documents/GitHub/Water-Level-Prediction/Data/test_data/Coastal Bend Met Measurements_2025-1.xlsx')

# Clean
datetime_col_water = [c for c in water_2025.columns if 'date' in c.lower() or 'time' in c.lower()][0]
datetime_col_met = [c for c in met_2025.columns if 'date' in c.lower() or 'time' in c.lower()][0]

water_2025['DateTime'] = pd.to_datetime(water_2025[datetime_col_water], errors='coerce')
water_2025 = water_2025[water_2025['DateTime'].notna()].copy()

met_2025['DateTime'] = pd.to_datetime(met_2025[datetime_col_met], errors='coerce')
met_2025 = met_2025[met_2025['DateTime'].notna()].copy()

print("\n[2025 Test Data Columns]")
print(f"\nWater Level ({len(water_2025.columns)} columns):")
print([c for c in water_2025.columns if c != 'DateTime'])

print(f"\nMeteorological ({len(met_2025.columns)} columns):")
print([c for c in met_2025.columns if c != 'DateTime'])

# Load training data to compare
print("\n[Training Data (2021-2024) Columns]")
train_data = pd.read_csv('../Data/modeling_enhanced/val_data.csv')

print(f"\nTraining ({len(train_data.columns)} columns):")
train_cols = [c for c in train_data.columns if c != 'datetime']
print(f"First 30: {train_cols[:30]}")

# Load required features
features_df = pd.read_csv('../Data/modeling_enhanced/features.csv')
all_training_features = features_df['feature'].tolist()

print("\n[Required Training Features]")
print(f"Total: {len(all_training_features)}")
print(all_training_features)

# Check which base columns are in training but not in 2025
print("\n[Missing Base Columns in 2025 Test Data]")

train_base_cols = [c for c in train_data.columns if c != 'datetime' and not any(x in c for x in ['lag', 'mean', 'std', 'rate', 'tide', 'lunar', 'wind', 'gradient', 'pressure', 'packery'])]
test_base_cols = list(water_2025.columns) + list(met_2025.columns)
test_base_cols = [c for c in test_base_cols if c not in ['DateTime', datetime_col_water, datetime_col_met]]

missing_base = [c for c in train_base_cols if c not in test_base_cols]

print(f"\nMissing {len(missing_base)} base columns:")
for col in missing_base:
    print(f"  - {col}")

# Categorize by station
print("\n[Missing by Station]")
for station in ['005', '008', '013', '202']:
    station_missing = [c for c in missing_base if c.startswith(f'{station}-')]
    if station_missing:
        print(f"\nStation {station} missing:")
        for col in station_missing:
            print(f"  - {col}")

# Check which meteorological variables are missing
print("\n[Meteorological Variables Analysis]")
met_vars = ['atp', 'wtp', 'wsd', 'wgt', 'wdr', 'bpr']

for var in met_vars:
    train_has = [c for c in train_base_cols if f'-{var}' in c]
    test_has = [c for c in test_base_cols if f'-{var}' in c]
    
    if len(train_has) != len(test_has):
        print(f"\n{var.upper()} mismatch:")
        print(f"  Training: {train_has}")
        print(f"  Test: {test_has}")
        missing = [c for c in train_has if c not in test_has]
        if missing:
            print(f"  Missing in test: {missing}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("\nOptions to fix:")
print("1. Get complete 2025 meteorological data with ALL variables")
print("2. Re-train models using ONLY features available in 2025 test data")
print("3. Use interpolation/nearest neighbor to fill missing stations")