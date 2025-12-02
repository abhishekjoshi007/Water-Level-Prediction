import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 2: ADVANCED FEATURE ENGINEERING")
print("="*80)

df = pd.read_csv('../Data/data_prepared/complete_merged_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"\n[1/10] Loaded: {df.shape}")

# ==============================================================================
# STATION SELECTION (Based on Quality Analysis)
# ==============================================================================

TARGET = '005-pwl'

wl_stations = ['005-pwl', '008-pwl', '013-pwl', '202-pwl']

met_features = {
    '005': ['atp', 'wtp', 'wsd', 'wgt', 'wdr', 'bpr'],
    '013': ['atp', 'wtp', 'wsd', 'wgt', 'wdr', 'bpr'],
    '202': ['atp', 'wtp', 'wsd', 'wgt', 'wdr', 'bpr'],
    '008': ['atp', 'bpr']
}

keep_cols = ['datetime'] + wl_stations
for station, vars in met_features.items():
    keep_cols.extend([f'{station}-{var}' for var in vars])

df = df[keep_cols].copy()

print(f"[2/10] Selected columns: {len(keep_cols)-1}")
print(f"  Water level: {len(wl_stations)}")
print(f"  Met features: {len(keep_cols) - len(wl_stations) - 1}")

# ==============================================================================
# FILL MISSING DATA
# ==============================================================================

print(f"\n[3/10] Filling missing data...")

for col in df.columns:
    if col != 'datetime':
        df[col] = df[col].interpolate(method='linear', limit=6)

df = df.fillna(method='ffill', limit=12)
df = df.fillna(method='bfill', limit=12)

initial_len = len(df)
df = df[df[TARGET].notna()]
print(f"  Rows: {initial_len:,} → {len(df):,}")

# ==============================================================================
# ASTRONOMICAL FEATURES
# ==============================================================================

print(f"\n[4/10] Creating astronomical features...")

epoch = pd.Timestamp('2000-01-01')
df['hours_since_epoch'] = (df['datetime'] - epoch).dt.total_seconds() / 3600

tidal_constituents = {
    'M2': 12.4206,
    'S2': 12.0000,
    'N2': 12.6583,
    'K1': 23.9345
}

for name, period in tidal_constituents.items():
    omega = 2 * np.pi / period
    df[f'tide_{name}_sin'] = np.sin(omega * df['hours_since_epoch'])
    df[f'tide_{name}_cos'] = np.cos(omega * df['hours_since_epoch'])

lunar_period = 29.53 * 24
df['lunar_phase'] = (df['hours_since_epoch'] % lunar_period) / lunar_period
df['spring_neap'] = np.cos(2 * np.pi * df['lunar_phase'])

print(f"  Added: 9 astronomical features")

# ==============================================================================
# WIND FEATURES
# ==============================================================================

print(f"\n[5/10] Creating wind features...")

for station in ['005', '013', '202']:
    if f'{station}-wsd' in df.columns and f'{station}-wdr' in df.columns:
        df[f'{station}_wind_u'] = df[f'{station}-wsd'] * np.cos(np.radians(df[f'{station}-wdr']))
        df[f'{station}_wind_v'] = df[f'{station}-wsd'] * np.sin(np.radians(df[f'{station}-wdr']))
        df[f'{station}_wind_stress'] = df[f'{station}-wsd'] ** 2

df['005_along_channel'] = df['005_wind_v']

print(f"  Added: 10 wind features")

# ==============================================================================
# PRESSURE FEATURES
# ==============================================================================

print(f"\n[6/10] Creating pressure features...")

df['pressure_change_1h'] = df['005-bpr'].diff(1)
df['pressure_change_3h'] = df['005-bpr'].diff(3)
df['pressure_change_6h'] = df['005-bpr'].diff(6)
df['pressure_accel'] = df['pressure_change_1h'].diff()

if '013-bpr' in df.columns and '202-bpr' in df.columns:
    df['pressure_gradient_NS'] = df['013-bpr'] - df['005-bpr']
    df['pressure_gradient_bay'] = df['202-bpr'] - df['005-bpr']

print(f"  Added: 6 pressure features")

# ==============================================================================
# TEMPERATURE FEATURES
# ==============================================================================

print(f"\n[7/10] Creating temperature features...")

if '005-wtp' in df.columns and '005-atp' in df.columns:
    df['temp_gradient_005'] = df['005-wtp'] - df['005-atp']

if '013-wtp' in df.columns:
    df['water_temp_diff_013'] = df['005-wtp'] - df['013-wtp']

if '202-wtp' in df.columns:
    df['water_temp_diff_202'] = df['005-wtp'] - df['202-wtp']

print(f"  Added: 3 temperature features")

# ==============================================================================
# SPATIAL FEATURES
# ==============================================================================

print(f"\n[8/10] Creating spatial features...")

df['gradient_NS'] = df['013-pwl'] - df['008-pwl']
df['gradient_bay_gulf'] = df['202-pwl'] - df['008-pwl']
df['gradient_NS_rate'] = df['gradient_NS'].diff()

spatial_stations = ['008-pwl', '013-pwl', '202-pwl']
df['spatial_mean'] = df[spatial_stations].mean(axis=1)
df['spatial_std'] = df[spatial_stations].std(axis=1)
df['packery_anomaly'] = df[TARGET] - df['spatial_mean']

print(f"  Added: 6 spatial features")

# ==============================================================================
# TEMPORAL FEATURES
# ==============================================================================

print(f"\n[9/10] Creating temporal features...")

for lag in [1, 2, 3, 6, 12, 24, 48]:
    df[f'packery_lag_{lag}h'] = df[TARGET].shift(lag)

for window in [6, 12, 24]:
    df[f'packery_mean_{window}h'] = df[TARGET].shift(1).rolling(window).mean()
    df[f'packery_std_{window}h'] = df[TARGET].shift(1).rolling(window).std()

for station in spatial_stations:
    df[f'{station}_lag1'] = df[station].shift(1)
    df[f'{station}_lag6'] = df[station].shift(6)

df['packery_rate_1h'] = df[TARGET].diff(1)
df['packery_rate_6h'] = df[TARGET].diff(6)

df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['day_of_year'] = df['datetime'].dt.dayofyear

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print(f"  Added: 29 temporal features")

# ==============================================================================
# FILL DERIVED FEATURES & CLEAN
# ==============================================================================

print(f"\n[10/10] Cleaning derived features...")

derived_cols = [col for col in df.columns if any(x in col for x in 
    ['lag', 'mean', 'std', 'rate', 'change', 'accel', 'gradient', 'diff'])]

for col in derived_cols:
    df[col] = df[col].fillna(method='ffill', limit=12)
    df[col] = df[col].fillna(method='bfill', limit=12)
    df[col] = df[col].fillna(0)

df = df.iloc[48:]

print(f"  Final rows: {len(df):,}")

# ==============================================================================
# FEATURE SELECTION & CORRELATION REMOVAL
# ==============================================================================

print(f"\n[Feature Selection]")

exclude_cols = ['datetime', TARGET, 'hours_since_epoch', 'lunar_phase', 
                'hour', 'day_of_week', 'month', 'day_of_year']
exclude_cols.extend([col for col in df.columns if '-' in col])

all_features = [col for col in df.columns if col not in exclude_cols]

print(f"  Total engineered: {len(all_features)}")

feature_df = df[all_features + [TARGET]].copy()
feature_df = feature_df.dropna()

corr_matrix = feature_df.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

if to_drop:
    print(f"  Removing {len(to_drop)} highly correlated (r>0.95)")
    all_features = [f for f in all_features if f not in to_drop]

print(f"  Final features: {len(all_features)}")

# ==============================================================================
# TRAIN/VALIDATION SPLIT
# ==============================================================================

print(f"\n[Splitting Data]")

train = df[df['datetime'] < '2024-01-01'].copy()
val = df[df['datetime'] >= '2024-01-01'].copy()

print(f"  Training:   {len(train):,} ({train['datetime'].min()} to {train['datetime'].max()})")
print(f"  Validation: {len(val):,} ({val['datetime'].min()} to {val['datetime'].max()})")

# ==============================================================================
# SAVE
# ==============================================================================

print(f"\n[Saving]")

import os
os.makedirs('../Data/modeling_enhanced', exist_ok=True)

train.to_csv('../Data/modeling_enhanced/train_data.csv', index=False)
val.to_csv('../Data/modeling_enhanced/val_data.csv', index=False)

pd.DataFrame({'feature': all_features}).to_csv('../Data/modeling_enhanced/features.csv', index=False)

metadata = {
    'target': TARGET,
    'n_features': len(all_features),
    'train_size': len(train),
    'val_size': len(val),
    'train_start': str(train['datetime'].min()),
    'train_end': str(train['datetime'].max()),
    'val_start': str(val['datetime'].min()),
    'val_end': str(val['datetime'].max())
}
pd.DataFrame([metadata]).to_csv('../Data/modeling_enhanced/metadata.csv', index=False)

print(f"  ✅ modeling_enhanced/train_data.csv")
print(f"  ✅ modeling_enhanced/val_data.csv")
print(f"  ✅ modeling_enhanced/features.csv")
print(f"  ✅ modeling_enhanced/metadata.csv")

# ==============================================================================
# FEATURE SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE")
print("="*80)

feature_groups = {
    'Astronomical': [f for f in all_features if 'tide' in f or 'lunar' in f or 'spring' in f],
    'Wind': [f for f in all_features if 'wind' in f],
    'Pressure': [f for f in all_features if 'pressure' in f],
    'Temperature': [f for f in all_features if 'temp' in f],
    'Spatial': [f for f in all_features if 'gradient' in f or 'spatial' in f or 'anomaly' in f],
    'Temporal Lags': [f for f in all_features if 'lag' in f],
    'Rolling Stats': [f for f in all_features if 'mean' in f or 'std' in f],
    'Rates': [f for f in all_features if 'rate' in f],
    'Time Encoding': [f for f in all_features if 'sin' in f or 'cos' in f]
}

print(f"\n📊 FEATURE GROUPS:")
for group, features in feature_groups.items():
    print(f"  {group:20s}: {len(features):2d} features")

print(f"\n🎯 READY FOR MODELING:")
print(f"  Features: {len(all_features)}")
print(f"  Training samples: {len(train):,}")
print(f"  Validation samples: {len(val):,}")

print("\n" + "="*80)
