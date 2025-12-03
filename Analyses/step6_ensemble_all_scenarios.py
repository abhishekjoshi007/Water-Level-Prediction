import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 6: ENSEMBLE FOR ALL SCENARIOS (FIXED GNN)")
print("="*80)

scenarios = ['Scenario1_1h', 'Scenario2_6h', 'Scenario3_12h']
horizons = {'Scenario1_1h': 1, 'Scenario2_6h': 6, 'Scenario3_12h': 12}

all_results = []

for scenario_name in scenarios:
    print("\n" + "="*80)
    print(f"{scenario_name} - ENSEMBLE")
    print("="*80)
    
    horizon = horizons[scenario_name]
    
    # Load all model predictions
    df_xgb = pd.read_csv(f'models/{scenario_name}/predictions.csv')
    df_lstm = pd.read_csv(f'models/{scenario_name}/lstm_predictions.csv')
    df_gnn = pd.read_csv(f'models/{scenario_name}/gnn_predictions_fixed.csv')
    
    # Align all predictions to same length
    min_length = min(len(df_xgb), len(df_lstm), len(df_gnn))
    
    actual = df_xgb['actual'].values[:min_length]
    persistence = df_xgb['persistence'].values[:min_length]
    datetime = df_xgb['datetime'].values[:min_length]
    
    xgb_pred = df_xgb['xgboost'].values[:min_length]
    rf_pred = df_xgb['random_forest'].values[:min_length]
    lstm_pred = df_lstm['lstm'].values[:min_length]
    gnn_pred = df_gnn['gnn'].values[:min_length]
    
    print(f"  Aligned samples: {len(actual):,}")
    
    # Create predictions matrix
    predictions_matrix = np.column_stack([xgb_pred, rf_pred, lstm_pred, gnn_pred])
    model_names = ['XGBoost', 'Random Forest', 'LSTM', 'GNN']
    
    # Simple average
    ensemble_simple = np.mean(predictions_matrix, axis=1)
    
    # Weighted optimization
    def weighted_mse(weights):
        weights = np.abs(weights)
        weights = weights / np.sum(weights)
        ensemble_pred = np.dot(predictions_matrix, weights)
        return mean_squared_error(actual, ensemble_pred)
    
    n_models = 4
    initial_weights = np.ones(n_models) / n_models
    result = minimize(weighted_mse, initial_weights, method='SLSQP', 
                     bounds=[(0, 1)] * n_models,
                     constraints={'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1})
    
    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / np.sum(optimal_weights)
    ensemble_weighted = np.dot(predictions_matrix, optimal_weights)
    
    print(f"\n  Optimal weights:")
    for name, weight in zip(model_names, optimal_weights):
        print(f"    {name:15s}: {weight:.3f}")
    
    # Stacking
    ridge = Ridge(alpha=1.0)
    ridge.fit(predictions_matrix, actual)
    ensemble_stacking = ridge.predict(predictions_matrix)
    
    # Calculate all metrics
    results_list = []
    
    # Persistence
    pers_residuals = actual - persistence
    results_list.append({
        'Scenario': scenario_name,
        'Model': 'Persistence',
        'Horizon_Hours': horizon,
        'Central Frequency 15cm': np.mean(np.abs(pers_residuals) <= 0.15) * 100,
        'MSE': mean_squared_error(actual, persistence),
        'RMSE': np.sqrt(mean_squared_error(actual, persistence)),
        'R2': r2_score(actual, persistence),
        'MAE': mean_absolute_error(actual, persistence),
        'MedAE': median_absolute_error(actual, persistence),
        'MaxErr': np.max(np.abs(pers_residuals)),
        'residual Stdev': np.std(pers_residuals)
    })
    
    # Individual models
    for name, pred in zip(model_names, [xgb_pred, rf_pred, lstm_pred, gnn_pred]):
        residuals = actual - pred
        results_list.append({
            'Scenario': scenario_name,
            'Model': name,
            'Horizon_Hours': horizon,
            'Central Frequency 15cm': np.mean(np.abs(residuals) <= 0.15) * 100,
            'MSE': mean_squared_error(actual, pred),
            'RMSE': np.sqrt(mean_squared_error(actual, pred)),
            'R2': r2_score(actual, pred),
            'MAE': mean_absolute_error(actual, pred),
            'MedAE': median_absolute_error(actual, pred),
            'MaxErr': np.max(np.abs(residuals)),
            'residual Stdev': np.std(residuals)
        })
    
    # Ensembles
    for ens_name, ens_pred in [('Ensemble_Simple', ensemble_simple), 
                                 ('Ensemble_Weighted', ensemble_weighted),
                                 ('Ensemble_Stacking', ensemble_stacking)]:
        residuals = actual - ens_pred
        results_list.append({
            'Scenario': scenario_name,
            'Model': ens_name,
            'Horizon_Hours': horizon,
            'Central Frequency 15cm': np.mean(np.abs(residuals) <= 0.15) * 100,
            'MSE': mean_squared_error(actual, ens_pred),
            'RMSE': np.sqrt(mean_squared_error(actual, ens_pred)),
            'R2': r2_score(actual, ens_pred),
            'MAE': mean_absolute_error(actual, ens_pred),
            'MedAE': median_absolute_error(actual, ens_pred),
            'MaxErr': np.max(np.abs(residuals)),
            'residual Stdev': np.std(residuals)
        })
    
    scenario_df = pd.DataFrame(results_list)
    print("\n" + scenario_df[['Model', 'RMSE', 'R2', 'Central Frequency 15cm']].to_string(index=False))
    
    scenario_df.to_csv(f'models/{scenario_name}/ensemble_results_fixed.csv', index=False)
    all_results.append(scenario_df)
    
    # Save all predictions
    pred_df = pd.DataFrame({
        'datetime': datetime,
        'actual': actual,
        'persistence': persistence,
        'xgboost': xgb_pred,
        'random_forest': rf_pred,
        'lstm': lstm_pred,
        'gnn': gnn_pred,
        'ensemble_simple': ensemble_simple,
        'ensemble_weighted': ensemble_weighted,
        'ensemble_stacking': ensemble_stacking
    })
    pred_df.to_csv(f'models/{scenario_name}/all_predictions_fixed.csv', index=False)
    
    print(f"\n  ✅ Saved: models/{scenario_name}/ensemble_results_fixed.csv")

print("\n" + "="*80)
print("COMBINED ENSEMBLE RESULTS")
print("="*80)

combined_df = pd.concat(all_results, ignore_index=True)
print("\n" + combined_df.to_string(index=False))

combined_df.to_csv('models/ensemble_all_scenarios_results_fixed.csv', index=False)
print(f"\n✅ Saved: models/ensemble_all_scenarios_results_fixed.csv")

print("\n" + "="*80)
print("BEST MODEL BY SCENARIO")
print("="*80)

for scenario in scenarios:
    scenario_data = combined_df[combined_df['Scenario'] == scenario]
    best_idx = scenario_data['RMSE'].idxmin()
    best_model = scenario_data.loc[best_idx]
    
    print(f"\n{scenario} ({best_model['Horizon_Hours']:.0f}h ahead):")
    print(f"  Best Model: {best_model['Model']}")
    print(f"  RMSE: {best_model['RMSE']:.4f} m ({best_model['RMSE']*100:.2f} cm)")
    print(f"  R²: {best_model['R2']:.4f}")
    print(f"  Within 15cm: {best_model['Central Frequency 15cm']:.1f}%")

print("\n" + "="*80)
print("✅ ALL MODELS COMPLETE!")
print("="*80)