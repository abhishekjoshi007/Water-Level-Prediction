import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("COMPREHENSIVE VISUALIZATION SUITE - FINAL VERSION")
print("="*80)

# ==============================================================================
# LOAD AND VALIDATE DATA
# ==============================================================================

print("\n[1/15] Loading and validating data...")

def load_and_validate(filepath, name):
    """Load CSV and validate data quality"""
    try:
        df = pd.read_csv(filepath)
        print(f"  ✅ {name}: {df.shape} - Columns: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"  ❌ Failed to load {name}: {e}")
        return None

# Load results
test_results = load_and_validate('models/2025_ALL_MODELS_comparison_FINAL.csv', 'Test Results')
val_results = load_and_validate('models/ensemble_all_scenarios_results_fixed.csv', 'Val Results')

scenarios = ['Scenario1_1h', 'Scenario2_6h', 'Scenario3_12h']
scenario_names = ['Scenario 1 (1h)', 'Scenario 2 (6h)', 'Scenario 3 (12h)']
scenario_names_simple = ['1h', '6h', '12h']

test_predictions = {}
val_predictions = {}

# Load prediction files
for scenario in scenarios:
    test_pred = load_and_validate(f'models/{scenario}/predictions_2025_ALL_MODELS_FINAL.csv', f'{scenario}_test')
    if test_pred is not None:
        test_pred['datetime'] = pd.to_datetime(test_pred['datetime'])
        test_predictions[scenario] = test_pred
    
    val_pred = load_and_validate(f'models/{scenario}/all_predictions_fixed.csv', f'{scenario}_val')
    if val_pred is not None:
        val_pred['datetime'] = pd.to_datetime(val_pred['datetime'])
        # Handle ensemble naming
        if 'ensemble' not in val_pred.columns and 'stacking' in val_pred.columns:
            val_pred['ensemble'] = val_pred['stacking']
        if 'ensemble' not in val_pred.columns and 'ensemble_stacking' in val_pred.columns:
            val_pred['ensemble'] = val_pred['ensemble_stacking']
        val_predictions[scenario] = val_pred

print(f"\n  📊 Loaded {len(test_predictions)} test & {len(val_predictions)} validation scenarios")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_model_data(data, model_name):
    """Get model predictions with fallback for different column names"""
    possible_names = [model_name]
    if model_name == 'ensemble':
        possible_names.extend(['stacking', 'ensemble_stacking', 'Ensemble', 'Stacking'])
    
    for name in possible_names:
        if name in data.columns:
            return data[name].values, name
    return None, None

def calculate_metrics(actual, predicted):
    """Calculate comprehensive metrics"""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    pred_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return None
    
    errors = (pred_clean - actual_clean) * 100
    
    return {
        'r2': r2_score(actual_clean, pred_clean),
        'rmse': np.sqrt(np.mean(errors**2)),
        'mae': np.mean(np.abs(errors)),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'within_15cm': np.mean(np.abs(errors) <= 15) * 100,
        'skew': pd.Series(errors).skew(),
        'n_samples': len(actual_clean)
    }

# ==============================================================================
# PLOT 1: Model Performance Comparison Bars (Validation vs Test)
# ==============================================================================

print("\n[2/15] Creating Plot 1: Model Performance Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (horizon, scenario) in enumerate(zip([1, 6, 12], scenarios)):
    ax = axes[idx]
    
    test_data = test_results[test_results['Scenario'] == scenario].iloc[0]
    val_data = val_results[val_results['Scenario'] == scenario]
    
    models = ['XGBoost', 'RF', 'LSTM', 'Ensemble']
    test_rmse = []
    val_rmse = []
    
    for model in models:
        test_rmse.append(test_data[f'{model}_RMSE'])
        
        val_model_name = 'Stacking' if model == 'Ensemble' else model
        val_model_data = val_data[val_data['Model'] == val_model_name]
        
        if len(val_model_data) > 0:
            val_rmse.append(val_model_data['RMSE'].values[0] * 100)
        else:
            val_rmse.append(np.nan)
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, val_rmse, width, label='Validation (2024)', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, test_rmse, width, label='Test (2025)', alpha=0.8, color='coral')
    
    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('RMSE (cm)', fontsize=11, fontweight='bold')
    ax.set_title(f'{horizon}h Ahead Predictions', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('plots/01_model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("  ✅ Saved: plots/01_model_performance_comparison.png")
plt.close()

# ==============================================================================
# PLOT 2: Performance Heatmap
# ==============================================================================

print("\n[3/15] Creating Plot 2: Performance Heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

val_pivot = val_results.pivot(index='Model', columns='Scenario', values='RMSE') * 100
sns.heatmap(val_pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[0], 
            cbar_kws={'label': 'RMSE (cm)'}, vmin=0, vmax=30)
axes[0].set_title('Validation (2024) - RMSE (cm)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Model', fontweight='bold')

test_data_for_heatmap = []
for scenario in scenarios:
    row_data = test_results[test_results['Scenario'] == scenario].iloc[0]
    for model in ['XGBoost', 'RF', 'LSTM', 'Ensemble']:
        test_data_for_heatmap.append({
            'Model': 'Stacking' if model == 'Ensemble' else model,
            'Scenario': scenario,
            'RMSE': row_data[f'{model}_RMSE']
        })

test_pivot = pd.DataFrame(test_data_for_heatmap).pivot(index='Model', columns='Scenario', values='RMSE')
sns.heatmap(test_pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[1], 
            cbar_kws={'label': 'RMSE (cm)'}, vmin=0, vmax=30)
axes[1].set_title('Test (2025) - RMSE (cm)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('plots/02_performance_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✅ Saved: plots/02_performance_heatmap.png")
plt.close()

# ==============================================================================
# PLOT 3 & 4: Time Series (Validation & Test)
# ==============================================================================

print("\n[4/15] Creating Plots 3-4: Time Series...")

for dataset_name, predictions, plot_num in [('Validation', val_predictions, '03'), ('Test 2025', test_predictions, '04')]:
    if len(predictions) > 0:
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            if scenario not in predictions:
                ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
                continue
                
            data = predictions[scenario].iloc[:168]  # First week
            
            ax.plot(data['datetime'], data['actual'], 'k-', linewidth=2, label='Actual', alpha=0.7)
            
            for model, style, label in [('xgboost', '--', 'XGBoost'), 
                                        ('random_forest', ':', 'Random Forest'),
                                        ('ensemble', '-', 'Ensemble')]:
                pred, _ = get_model_data(data, model)
                if pred is not None:
                    ax.plot(data['datetime'], pred, style, linewidth=1.5, label=label, alpha=0.8)
            
            ax.set_ylabel('Water Level (m)', fontweight='bold')
            ax.set_title(f'{dataset_name} - {scenario.replace("_", " ")} (First Week)', 
                        fontsize=12, fontweight='bold')
            ax.legend(loc='best', ncol=4)
            ax.grid(alpha=0.3)
            
            if idx == 2:
                ax.set_xlabel('Date', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'plots/{plot_num}_{dataset_name.lower().replace(" ", "")}_timeseries_week.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: plots/{plot_num}_{dataset_name.lower().replace(' ', '')}_timeseries_week.png")
        plt.close()

# ==============================================================================
# PLOT 5 & 6: COMPLETE SCATTER PLOTS
# ==============================================================================

print("\n[5/15] Creating Plots 5-6: Complete Scatter Plots...")

for dataset_name, predictions, color, plot_num in [('Validation', val_predictions, 'steelblue', '05'), 
                                                     ('Test 2025', test_predictions, 'coral', '06')]:
    if len(predictions) == 0:
        continue
        
    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    
    models = ['xgboost', 'random_forest', 'lstm', 'ensemble']
    model_names = ['XGBoost', 'Random Forest', 'LSTM', 'Ensemble']
    
    for row_idx, (scenario, scenario_name) in enumerate(zip(scenarios, scenario_names)):
        if scenario not in predictions:
            continue
            
        data = predictions[scenario]
        actual = data['actual'].values
        
        for col_idx, (model, model_name) in enumerate(zip(models, model_names)):
            ax = axes[row_idx, col_idx]
            
            predictions_data, col_name = get_model_data(data, model)
            
            if predictions_data is None:
                ax.text(0.5, 0.5, 'Model Not Available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
                if row_idx == 0:
                    ax.set_title(model_name, fontweight='bold', fontsize=13)
                continue
            
            metrics = calculate_metrics(actual, predictions_data)
            if metrics is None:
                continue
            
            n_samples = min(2000, len(data))
            sample_idx = np.random.choice(len(data), size=n_samples, replace=False)
            actual_sample = actual[sample_idx]
            pred_sample = predictions_data[sample_idx]
            
            ax.scatter(actual_sample, pred_sample, alpha=0.4, s=20, c=color, edgecolors='none')
            
            min_val = min(actual_sample.min(), pred_sample.min())
            max_val = max(actual_sample.max(), pred_sample.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect', zorder=5)
            
            metrics_text = (f"R² = {metrics['r2']:.3f}\n"
                          f"RMSE = {metrics['rmse']:.1f} cm\n"
                          f"MAE = {metrics['mae']:.1f} cm\n"
                          f"Within ±15cm: {metrics['within_15cm']:.1f}%")
            
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                            edgecolor='black', linewidth=1),
                   fontsize=9, family='monospace')
            
            if row_idx == 0:
                ax.set_title(model_name, fontweight='bold', fontsize=13, pad=10)
            if col_idx == 0:
                ax.set_ylabel(f'{scenario_name}\nPredicted (m)', fontweight='bold', fontsize=11)
            if row_idx == 2:
                ax.set_xlabel('Actual Water Level (m)', fontweight='bold', fontsize=11)
            
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_aspect('equal', adjustable='box')
    
    title = 'Validation (2024)' if dataset_name == 'Validation' else 'Test 2025'
    plt.suptitle(f'{title} - Actual vs Predicted Water Levels', fontsize=18, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig(f'plots/{plot_num}_{dataset_name.lower().replace(" ", "")}_scatter_complete.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: plots/{plot_num}_{dataset_name.lower().replace(' ', '')}_scatter_complete.png")
    plt.close()

# ==============================================================================
# PLOT 7 & 8: ERROR DISTRIBUTIONS
# ==============================================================================

print("\n[6/15] Creating Plots 7-8: Error Distributions...")

for dataset_name, predictions, color, plot_num in [('Validation', val_predictions, 'steelblue', '07'), 
                                                     ('Test 2025', test_predictions, 'coral', '08')]:
    if len(predictions) == 0:
        continue
        
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    
    models = ['xgboost', 'random_forest', 'lstm', 'ensemble']
    model_names = ['XGBoost', 'Random Forest', 'LSTM', 'Ensemble']
    
    for row_idx, (scenario, scenario_name) in enumerate(zip(scenarios, scenario_names)):
        if scenario not in predictions:
            continue
            
        data = predictions[scenario]
        actual = data['actual'].values
        
        for col_idx, (model, model_name) in enumerate(zip(models, model_names)):
            ax = axes[row_idx, col_idx]
            
            predictions_data, col_name = get_model_data(data, model)
            
            if predictions_data is None:
                ax.text(0.5, 0.5, 'Model Not Available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='red')
                if row_idx == 0:
                    ax.set_title(model_name, fontweight='bold', fontsize=13)
                continue
            
            errors = (predictions_data - actual) * 100
            errors = errors[~np.isnan(errors)]
            
            if len(errors) == 0:
                continue
            
            n_bins = min(80, max(30, len(errors) // 100))
            ax.hist(errors, bins=n_bins, alpha=0.7, edgecolor='black', linewidth=0.5, color=color)
            
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Zero Error', zorder=5)
            ax.axvline(x=errors.mean(), color='blue', linestyle='--', linewidth=2, 
                      label=f'Mean: {errors.mean():.1f}cm', zorder=5)
            ax.axvline(x=-15, color='orange', linestyle=':', linewidth=2.5, alpha=0.8)
            ax.axvline(x=15, color='orange', linestyle=':', linewidth=2.5, alpha=0.8, label='±15cm threshold')
            
            within_15 = np.mean(np.abs(errors) <= 15) * 100
            std_error = errors.std()
            skewness = pd.Series(errors).skew()
            
            stats_text = (f"Within ±15cm: {within_15:.1f}%\n"
                         f"Std Dev: {std_error:.1f} cm\n"
                         f"Skewness: {skewness:.2f}\n"
                         f"N = {len(errors):,}")
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                            edgecolor='black', linewidth=1),
                   fontsize=9, family='monospace')
            
            if row_idx == 0:
                ax.set_title(model_name, fontweight='bold', fontsize=13, pad=10)
            if col_idx == 0:
                ax.set_ylabel(f'{scenario_name}\nFrequency', fontweight='bold', fontsize=11)
            if row_idx == 2:
                ax.set_xlabel('Prediction Error (cm)', fontweight='bold', fontsize=11)
            
            if row_idx == 0 and col_idx == 3:
                ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
            
            ax.grid(alpha=0.3, axis='y', linestyle='--')
            xlim_max = min(np.percentile(np.abs(errors), 99.5), 100)
            ax.set_xlim(-xlim_max, xlim_max)
    
    title = 'Validation (2024)' if dataset_name == 'Validation' else 'Test 2025'
    plt.suptitle(f'{title} - Prediction Error Distribution', fontsize=18, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig(f'plots/{plot_num}_{dataset_name.lower().replace(" ", "")}_error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: plots/{plot_num}_{dataset_name.lower().replace(' ', '')}_error_distribution.png")
    plt.close()

# ==============================================================================
# PLOT 9: Validation vs Test Comparison (4 subplots)
# ==============================================================================

print("\n[7/15] Creating Plot 9: Validation vs Test Comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models = ['XGBoost', 'RF', 'LSTM', 'Ensemble']

# RMSE Comparison
ax = axes[0, 0]
for model in models:
    val_rmse = []
    test_rmse = []
    
    for scenario in scenarios:
        test_data = test_results[test_results['Scenario'] == scenario].iloc[0]
        test_rmse.append(test_data[f'{model}_RMSE'])
        
        val_model_name = 'Stacking' if model == 'Ensemble' else model
        val_data = val_results[(val_results['Scenario'] == scenario) & 
                              (val_results['Model'] == val_model_name)]
        if len(val_data) > 0:
            val_rmse.append(val_data['RMSE'].values[0] * 100)
        else:
            val_rmse.append(np.nan)
    
    ax.plot(scenario_names_simple, val_rmse, 'o-', label=f'{model} (Val)', linewidth=2, markersize=8)
    ax.plot(scenario_names_simple, test_rmse, 's--', label=f'{model} (Test)', linewidth=2, markersize=8, alpha=0.7)

ax.set_xlabel('Prediction Horizon', fontweight='bold', fontsize=11)
ax.set_ylabel('RMSE (cm)', fontweight='bold', fontsize=11)
ax.set_title('RMSE: Validation vs Test', fontweight='bold', fontsize=12)
ax.legend(ncol=2, fontsize=8, loc='upper left')
ax.grid(alpha=0.3)

# R² Comparison
ax = axes[0, 1]
for model in models:
    val_r2 = []
    test_r2 = []
    
    for scenario in scenarios:
        test_data = test_results[test_results['Scenario'] == scenario].iloc[0]
        test_r2.append(test_data[f'{model}_R2'])
        
        val_model_name = 'Stacking' if model == 'Ensemble' else model
        val_data = val_results[(val_results['Scenario'] == scenario) & 
                              (val_results['Model'] == val_model_name)]
        if len(val_data) > 0:
            val_r2.append(val_data['R2'].values[0])
        else:
            val_r2.append(np.nan)
    
    ax.plot(scenario_names_simple, val_r2, 'o-', label=f'{model} (Val)', linewidth=2, markersize=8)
    ax.plot(scenario_names_simple, test_r2, 's--', label=f'{model} (Test)', linewidth=2, markersize=8, alpha=0.7)

ax.set_xlabel('Prediction Horizon', fontweight='bold', fontsize=11)
ax.set_ylabel('R²', fontweight='bold', fontsize=11)
ax.set_title('R²: Validation vs Test', fontweight='bold', fontsize=12)
ax.axhline(y=0, color='red', linestyle=':', alpha=0.5)
ax.legend(ncol=2, fontsize=8, loc='lower left')
ax.grid(alpha=0.3)

# Within 15cm threshold
ax = axes[1, 0]
for model in models:
    test_within15 = []
    for scenario in scenarios:
        test_data = test_results[test_results['Scenario'] == scenario].iloc[0]
        test_within15.append(test_data[f'{model}_Within15'])
    ax.plot(scenario_names_simple, test_within15, 's-', label=model, linewidth=2, markersize=8)

ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
ax.set_xlabel('Prediction Horizon', fontweight='bold', fontsize=11)
ax.set_ylabel('% Predictions Within 15cm', fontweight='bold', fontsize=11)
ax.set_title('Operational Reliability (Test 2025)', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim([0, 105])

# Improvement over persistence
ax = axes[1, 1]
for model in models:
    test_imp = []
    for scenario in scenarios:
        test_data = test_results[test_results['Scenario'] == scenario].iloc[0]
        test_imp.append(test_data[f'{model}_Imp'])
    ax.plot(scenario_names_simple, test_imp, 's-', label=model, linewidth=2, markersize=8)

ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Persistence baseline')
ax.set_xlabel('Prediction Horizon', fontweight='bold', fontsize=11)
ax.set_ylabel('Improvement over Persistence (%)', fontweight='bold', fontsize=11)
ax.set_title('Model Improvement (Test 2025)', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/09_validation_vs_test_comparison.png', dpi=300, bbox_inches='tight')
print("  ✅ Saved: plots/09_validation_vs_test_comparison.png")
plt.close()

# ==============================================================================
# PLOT 10: Summary Performance Table
# ==============================================================================

print("\n[8/15] Creating Plot 10: Summary Performance Table...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

summary_data = []
for scenario in scenarios:
    test_data = test_results[test_results['Scenario'] == scenario].iloc[0]
    row = [
        scenario.replace('_', ' '),
        f"{test_data['Persistence_RMSE']:.2f}",
        f"{test_data['XGBoost_RMSE']:.2f}",
        f"{test_data['RF_RMSE']:.2f}",
        f"{test_data['LSTM_RMSE']:.2f}",
        f"{test_data['Ensemble_RMSE']:.2f}",
    ]
    summary_data.append(row)

columns = ['Scenario', 'Persistence', 'XGBoost', 'Random Forest', 'LSTM', 'Ensemble']

table = ax.table(cellText=summary_data, colLabels=columns, cellLoc='center', loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for i in range(len(columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

for row_idx in range(len(summary_data)):
    values = [float(summary_data[row_idx][i]) for i in range(1, 6)]
    min_val = min(values)
    min_col = values.index(min_val) + 1
    table[(row_idx + 1, min_col)].set_facecolor('#FFD700')
    table[(row_idx + 1, min_col)].set_text_props(weight='bold')

plt.title('Test 2025 - RMSE Summary (cm)', fontsize=14, fontweight='bold', pad=20)
plt.savefig('plots/10_summary_performance_table.png', dpi=300, bbox_inches='tight')
print("  ✅ Saved: plots/10_summary_performance_table.png")
plt.close()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("✅ COMPREHENSIVE VISUALIZATION SUITE COMPLETE!")
print("="*80)
print("\n📊 Created 10 plot sets (15 total files):")
print("   01. Model Performance Comparison Bars")
print("   02. Performance Heatmap (Val vs Test)")
print("   03. Validation Time Series")
print("   04. Test 2025 Time Series")
print("   05. Validation Scatter Plots (All 4 models × 3 scenarios)")
print("   06. Test 2025 Scatter Plots (All 4 models × 3 scenarios)")
print("   07. Validation Error Distributions (All models)")
print("   08. Test 2025 Error Distributions (All models)")
print("   09. Validation vs Test 4-Panel Comparison")
print("   10. Summary Performance Table")
print("\n💾 All plots saved in: plots/")
print("\n🎯 Key Features:")
print("   • Robust data loading with fallback for column names")
print("   • Comprehensive metrics on all plots")
print("   • Publication-quality styling")
print("   • Handles missing models gracefully")
print("   • Equal aspect ratios on scatter plots")
print("   • Proper error distributions with statistics")
print("="*80)