import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 1: COMPREHENSIVE DATA QUALITY ANALYSIS")
print("="*80)

# LOAD ALL DATA (NO FILTERING)

print("\n[1/8] Loading ALL data files...")

# Water level data - ALL STATIONS
df_wl = pd.read_excel("../Data/Coastal Bend Water Level Measurements_2021-2024.xls")
df_wl = df_wl.rename(columns={df_wl.columns[0]: 'datetime'})
df_wl['datetime'] = pd.to_datetime(df_wl['datetime'])

print(f"  ✅ Water level data: {df_wl.shape}")
print(f"     Columns: {df_wl.columns.tolist()}")

# Meteorological data - ALL STATIONS, ALL VARIABLES
df_met = pd.read_excel("../Data/Coastal Bend Met Measurements_2021-2024.xls")
df_met = df_met.rename(columns={df_met.columns[0]: 'datetime'})
df_met['datetime'] = pd.to_datetime(df_met['datetime'])

print(f"  ✅ Met data: {df_met.shape}")
print(f"     Sample columns: {df_met.columns.tolist()[:10]}")

# MERGE ALL DATA

print("\n[2/8] Merging ALL datasets...")

df_complete = pd.merge(df_wl, df_met, on='datetime', how='inner')
print(f"  ✅ Complete merged data: {df_complete.shape}")
print(f"  Total columns: {len(df_complete.columns)} (1 datetime + {len(df_complete.columns)-1} data columns)")
print(f"  Date range: {df_complete['datetime'].min()} to {df_complete['datetime'].max()}")
print(f"  Total hours: {len(df_complete):,}")

# COMPREHENSIVE DATA QUALITY ANALYSIS - WATER LEVEL

print("\n[3/8] Water Level Data Quality Analysis...")

wl_stations = [col for col in df_complete.columns if '-pwl' in col]
print(f"\n  Total Water Level Stations: {len(wl_stations)}")
print(f"  " + "="*60)

wl_quality = []

for station in wl_stations:
    station_id = station.split('-')[0]
    
    # Missing data
    total_records = len(df_complete)
    missing_count = df_complete[station].isna().sum()
    missing_pct = (missing_count / total_records) * 100
    
    # Valid data statistics
    valid_data = df_complete[station].dropna()
    
    if len(valid_data) > 0:
        mean_val = valid_data.mean()
        std_val = valid_data.std()
        min_val = valid_data.min()
        max_val = valid_data.max()
        
        # Consecutive missing periods
        is_missing = df_complete[station].isna()
        missing_groups = (is_missing != is_missing.shift()).cumsum()
        max_consecutive_missing = is_missing.groupby(missing_groups).sum().max()
    else:
        mean_val = std_val = min_val = max_val = np.nan
        max_consecutive_missing = total_records
    
    # Quality classification
    if missing_pct < 5:
        quality = "✅ Excellent"
    elif missing_pct < 10:
        quality = "✅ Good"
    elif missing_pct < 20:
        quality = "⚠️  Acceptable"
    elif missing_pct < 50:
        quality = "⚠️  Poor"
    else:
        quality = "❌ Unusable"
    
    wl_quality.append({
        'Station': station,
        'ID': station_id,
        'Missing_Count': missing_count,
        'Missing_Pct': missing_pct,
        'Valid_Records': total_records - missing_count,
        'Mean': mean_val,
        'Std': std_val,
        'Min': min_val,
        'Max': max_val,
        'Max_Gap_Hours': int(max_consecutive_missing),
        'Quality': quality
    })
    
    print(f"\n  {station} (Station {station_id}):")
    print(f"    Missing: {missing_count:,} / {total_records:,} ({missing_pct:.1f}%)")
    print(f"    Valid records: {total_records - missing_count:,}")
    print(f"    Range: {min_val:.3f} to {max_val:.3f} m" if not np.isnan(min_val) else "    Range: No valid data")
    print(f"    Mean ± Std: {mean_val:.3f} ± {std_val:.3f} m" if not np.isnan(mean_val) else "    Mean ± Std: N/A")
    print(f"    Max consecutive gap: {int(max_consecutive_missing)} hours")
    print(f"    Quality: {quality}")

# Create DataFrame for easy analysis
df_wl_quality = pd.DataFrame(wl_quality)

# COMPREHENSIVE DATA QUALITY ANALYSIS - METEOROLOGICAL

print("\n[4/8] Meteorological Data Quality Analysis...")

# Group by variable type
met_variables = {
    'atp': 'Air Temperature (°C)',
    'wtp': 'Water Temperature (°C)',
    'wsd': 'Wind Speed (m/s)',
    'wgt': 'Wind Gust (m/s)',
    'wdr': 'Wind Direction (°)',
    'bpr': 'Barometric Pressure (mb)'
}

met_quality = []

for var_code, var_name in met_variables.items():
    var_cols = [col for col in df_complete.columns if f'-{var_code}' in col]
    
    print(f"\n  {var_name} ({var_code})")
    print(f"  " + "-"*60)
    print(f"  Available at {len(var_cols)} stations")
    
    for col in var_cols:
        station_id = col.split('-')[0]
        
        total_records = len(df_complete)
        missing_count = df_complete[col].isna().sum()
        missing_pct = (missing_count / total_records) * 100
        
        valid_data = df_complete[col].dropna()
        
        if len(valid_data) > 0:
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            min_val = valid_data.min()
            max_val = valid_data.max()
        else:
            mean_val = std_val = min_val = max_val = np.nan
        
        # Quality classification
        if missing_pct < 5:
            quality = "✅ Excellent"
        elif missing_pct < 10:
            quality = "✅ Good"
        elif missing_pct < 20:
            quality = "⚠️  Acceptable"
        elif missing_pct < 50:
            quality = "⚠️  Poor"
        else:
            quality = "❌ Unusable"
        
        met_quality.append({
            'Column': col,
            'Station': station_id,
            'Variable': var_code,
            'Variable_Name': var_name,
            'Missing_Count': missing_count,
            'Missing_Pct': missing_pct,
            'Valid_Records': total_records - missing_count,
            'Mean': mean_val,
            'Std': std_val,
            'Min': min_val,
            'Max': max_val,
            'Quality': quality
        })
        
        print(f"    Station {station_id}: {missing_pct:5.1f}% missing - {quality}")

df_met_quality = pd.DataFrame(met_quality)

# STATION-BY-STATION SUMMARY

print("\n[5/8] Station-by-Station Complete Summary...")

all_station_ids = sorted(set([col.split('-')[0] for col in df_complete.columns if '-' in col and col != 'datetime']))

print(f"\n  Total Stations: {len(all_station_ids)}")
print(f"  " + "="*80)

station_summary = []

for station_id in all_station_ids:
    print(f"\n  📍 STATION {station_id}:")
    
    # Get all columns for this station
    station_cols = [col for col in df_complete.columns if col.startswith(f'{station_id}-')]
    
    print(f"    Available measurements: {len(station_cols)}")
    
    # Count by type
    has_pwl = any('-pwl' in col for col in station_cols)
    has_met = len([col for col in station_cols if '-pwl' not in col])
    
    print(f"    - Water Level (pwl): {'✅ Yes' if has_pwl else '❌ No'}")
    print(f"    - Meteorological: {has_met} variables")
    
    # Overall quality
    station_data = df_complete[station_cols]
    total_missing = station_data.isna().sum().sum()
    total_cells = len(station_data) * len(station_cols)
    overall_missing_pct = (total_missing / total_cells) * 100
    
    print(f"    Overall missing: {overall_missing_pct:.1f}%")
    
    # Individual variables
    print(f"    Variables:")
    for col in sorted(station_cols):
        var = col.split('-')[1]
        missing_pct = (df_complete[col].isna().sum() / len(df_complete)) * 100
        status = "✅" if missing_pct < 10 else "⚠️" if missing_pct < 30 else "❌"
        print(f"      {var}: {missing_pct:5.1f}% missing {status}")
    
    # Recommendation
    if overall_missing_pct < 10:
        recommendation = "✅ RECOMMENDED for modeling"
    elif overall_missing_pct < 20:
        recommendation = "⚠️  USE WITH CAUTION"
    else:
        recommendation = "❌ NOT RECOMMENDED (too much missing data)"
    
    print(f"    Recommendation: {recommendation}")
    
    station_summary.append({
        'Station': station_id,
        'Has_WL': has_pwl,
        'Met_Variables': has_met,
        'Total_Columns': len(station_cols),
        'Overall_Missing_Pct': overall_missing_pct,
        'Recommendation': recommendation
    })

df_station_summary = pd.DataFrame(station_summary)

# TEMPORAL COVERAGE ANALYSIS

print("\n[6/8] Temporal Coverage Analysis...")

# Check for time gaps
df_complete['time_diff'] = df_complete['datetime'].diff()
gaps = df_complete[df_complete['time_diff'] > pd.Timedelta(hours=1)]

print(f"\n  Expected hourly frequency: 1 hour between records")
print(f"  Total records: {len(df_complete):,}")
print(f"  Date range: {df_complete['datetime'].min()} to {df_complete['datetime'].max()}")

date_range = (df_complete['datetime'].max() - df_complete['datetime'].min()).days
expected_records = date_range * 24

print(f"  Expected records (based on date range): {expected_records:,}")
print(f"  Actual records: {len(df_complete):,}")
print(f"  Coverage: {len(df_complete)/expected_records*100:.1f}%")

if len(gaps) > 0:
    print(f"\n  ⚠️  Found {len(gaps)} time gaps > 1 hour:")
    print(f"     Largest gap: {gaps['time_diff'].max()}")
    print(f"     Gaps > 24 hours: {len(gaps[gaps['time_diff'] > pd.Timedelta(hours=24)])}")
else:
    print(f"\n  ✅ No significant time gaps detected")

# Monthly coverage
monthly_coverage = df_complete.set_index('datetime').resample('M').size()
print(f"\n  Monthly record counts:")
print(f"    Min: {monthly_coverage.min()} records/month")
print(f"    Max: {monthly_coverage.max()} records/month")
print(f"    Mean: {monthly_coverage.mean():.0f} records/month")
print(f"    Expected: ~720 records/month (30 days × 24 hours)")

# CORRELATION ANALYSIS (Only for good quality stations)

print("\n[7/8] Correlation Analysis...")

# Select stations with < 15% missing for correlation
good_wl_stations = df_wl_quality[df_wl_quality['Missing_Pct'] < 15]['Station'].tolist()

if len(good_wl_stations) > 1:
    print(f"\n  Analyzing correlations for {len(good_wl_stations)} good quality stations:")
    for station in good_wl_stations:
        print(f"    {station}")
    
    # Calculate correlations
    corr_matrix = df_complete[good_wl_stations].corr()
    
    print(f"\n  Correlation Matrix:")
    print(corr_matrix.to_string())
    
    # Find highly correlated pairs
    print(f"\n  Highly correlated station pairs (r > 0.8):")
    for i in range(len(good_wl_stations)):
        for j in range(i+1, len(good_wl_stations)):
            r = corr_matrix.iloc[i, j]
            if r > 0.8:
                print(f"    {good_wl_stations[i]} ↔ {good_wl_stations[j]}: r = {r:.3f}")
else:
    print(f"  ⚠️  Not enough good quality stations for correlation analysis")

# SAVE COMPLETE DATA & QUALITY REPORTS

print("\n[8/8] Saving complete data and quality reports...")

import os
os.makedirs('data_prepared', exist_ok=True)

# Save complete merged data (NO FILTERING)
df_complete.to_csv('data_prepared/complete_merged_data.csv', index=False)
print(f"  ✅ Saved: data_prepared/complete_merged_data.csv")
print(f"     ({len(df_complete):,} rows × {len(df_complete.columns)} columns)")

# Save quality reports
df_wl_quality.to_csv('data_prepared/water_level_quality_report.csv', index=False)
print(f"  ✅ Saved: data_prepared/water_level_quality_report.csv")

df_met_quality.to_csv('data_prepared/meteorological_quality_report.csv', index=False)
print(f"  ✅ Saved: data_prepared/meteorological_quality_report.csv")

df_station_summary.to_csv('data_prepared/station_summary.csv', index=False)
print(f"  ✅ Saved: data_prepared/station_summary.csv")

# COMPREHENSIVE VISUALIZATIONS

print("\n[Visualization] Creating comprehensive data quality visualizations...")

fig = plt.figure(figsize=(20, 12))

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Missing data by station (Water Level)
ax1 = fig.add_subplot(gs[0, :])
stations_sorted = df_wl_quality.sort_values('Missing_Pct')
colors = ['green' if x < 10 else 'orange' if x < 30 else 'red' 
          for x in stations_sorted['Missing_Pct']]
ax1.barh(range(len(stations_sorted)), stations_sorted['Missing_Pct'], color=colors)
ax1.set_yticks(range(len(stations_sorted)))
ax1.set_yticklabels(stations_sorted['Station'])
ax1.set_xlabel('Missing Data (%)', fontsize=11)
ax1.set_title('Water Level Stations - Data Quality', fontsize=13, fontweight='bold')
ax1.axvline(10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
ax1.axvline(30, color='red', linestyle='--', alpha=0.5, label='30% threshold')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Meteorological variables availability heatmap
ax2 = fig.add_subplot(gs[1, :2])
met_pivot = df_met_quality.pivot_table(
    values='Missing_Pct', 
    index='Station', 
    columns='Variable',
    aggfunc='mean'
)
sns.heatmap(met_pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', 
            center=20, vmin=0, vmax=100, ax=ax2, cbar_kws={'label': 'Missing %'})
ax2.set_title('Meteorological Data Availability Heatmap', fontsize=13, fontweight='bold')
ax2.set_xlabel('Variable')
ax2.set_ylabel('Station')

# Plot 3: Station summary
ax3 = fig.add_subplot(gs[1, 2])
summary_counts = df_station_summary['Recommendation'].value_counts()
colors_summary = {'✅ RECOMMENDED for modeling': 'green',
                  '⚠️  USE WITH CAUTION': 'orange',
                  '❌ NOT RECOMMENDED (too much missing data)': 'red'}
ax3.pie(summary_counts, labels=summary_counts.index, autopct='%1.0f%%',
        colors=[colors_summary.get(x, 'gray') for x in summary_counts.index])
ax3.set_title('Station Quality Summary', fontsize=13, fontweight='bold')

# Plot 4: Temporal coverage
ax4 = fig.add_subplot(gs[2, 0])
monthly = df_complete.set_index('datetime').resample('M').size()
ax4.plot(monthly.index, monthly.values, marker='o', linewidth=2)
ax4.axhline(720, color='red', linestyle='--', alpha=0.5, label='Expected (~720/month)')
ax4.set_title('Temporal Coverage', fontsize=13, fontweight='bold')
ax4.set_ylabel('Records per Month')
ax4.legend()
ax4.grid(alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

# Plot 5: Correlation heatmap (good stations only)
ax5 = fig.add_subplot(gs[2, 1])
if len(good_wl_stations) > 1:
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, ax=ax5)
    ax5.set_title('Station Correlations\n(Good Quality Only)', fontsize=13, fontweight='bold')
else:
    ax5.text(0.5, 0.5, 'Not enough\ngood quality stations', 
             ha='center', va='center', fontsize=12)
    ax5.set_title('Station Correlations', fontsize=13, fontweight='bold')
    ax5.axis('off')

# Plot 6: Time series sample
ax6 = fig.add_subplot(gs[2, 2])
sample_data = df_complete.iloc[:168]  # First week
for station in good_wl_stations[:4]:  # Plot up to 4 stations
    ax6.plot(sample_data['datetime'], sample_data[station], 
             label=station, alpha=0.7, linewidth=1.5)
ax6.set_title('Sample Time Series\n(First Week)', fontsize=13, fontweight='bold')
ax6.set_ylabel('Water Level (m)')
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)
ax6.tick_params(axis='x', rotation=45)

plt.savefig('data_prepared/comprehensive_data_quality.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: data_prepared/comprehensive_data_quality.png")

plt.close()

# DETAILED SUMMARY & RECOMMENDATIONS

print("\n" + "="*80)
print("COMPREHENSIVE DATA QUALITY SUMMARY")
print("="*80)

print(f"\n📊 DATASET OVERVIEW:")
print(f"  Total records: {len(df_complete):,}")
print(f"  Date range: {df_complete['datetime'].min()} to {df_complete['datetime'].max()}")
print(f"  Total columns: {len(df_complete.columns)}")
print(f"    - Water level stations: {len(wl_stations)}")
print(f"    - Meteorological columns: {len(df_met_quality)}")

print(f"\n🏆 RECOMMENDED STATIONS (< 10% missing):")
recommended = df_wl_quality[df_wl_quality['Missing_Pct'] < 10]
if len(recommended) > 0:
    for _, row in recommended.iterrows():
        print(f"  ✅ {row['Station']}: {row['Missing_Pct']:.1f}% missing")
else:
    print(f"  ⚠️  No stations with < 10% missing")

print(f"\n⚠️  CAUTION STATIONS (10-30% missing):")
caution = df_wl_quality[(df_wl_quality['Missing_Pct'] >= 10) & (df_wl_quality['Missing_Pct'] < 30)]
if len(caution) > 0:
    for _, row in caution.iterrows():
        print(f"  ⚠️  {row['Station']}: {row['Missing_Pct']:.1f}% missing")
else:
    print(f"  (None)")

print(f"\n❌ NOT RECOMMENDED (> 30% missing):")
not_recommended = df_wl_quality[df_wl_quality['Missing_Pct'] >= 30]
if len(not_recommended) > 0:
    for _, row in not_recommended.iterrows():
        print(f"  ❌ {row['Station']}: {row['Missing_Pct']:.1f}% missing")
else:
    print(f"  (None)")

print(f"\n📋 AVAILABLE METEOROLOGICAL VARIABLES:")
for var_code, var_name in met_variables.items():
    var_data = df_met_quality[df_met_quality['Variable'] == var_code]
    available_stations = len(var_data)
    good_quality = len(var_data[var_data['Missing_Pct'] < 10])
    print(f"  {var_code} ({var_name}):")
    print(f"    Available at: {available_stations} stations")
    print(f"    Good quality: {good_quality} stations (< 10% missing)")

print("\n" + "="*80)
print("✅ STEP 1 COMPLETE - COMPREHENSIVE DATA QUALITY CHECK DONE!")
print("="*80)

print(f"\n📁 OUTPUT FILES:")
print(f"  1. data_prepared/complete_merged_data.csv - ALL data (no filtering)")
print(f"  2. data_prepared/water_level_quality_report.csv - WL quality metrics")
print(f"  3. data_prepared/meteorological_quality_report.csv - Met quality metrics")
print(f"  4. data_prepared/station_summary.csv - Station-by-station summary")
print(f"  5. data_prepared/comprehensive_data_quality.png - Visual quality report")

