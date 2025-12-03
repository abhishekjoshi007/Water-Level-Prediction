import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE 2025 TEST - HYBRID NaN FILLING")
print("="*80)

# LOAD AND PREPARE 2025 DATA

print("\n[1/5] Loading 2025 Test Data...")

water_2025_raw = pd.read_excel('/Users/abhishekjoshi/Documents/GitHub/Water-Level-Prediction/Data/test_data/Coastal Bend Water Level Measurements_2025.xlsx')
met_2025_raw = pd.read_excel('/Users/abhishekjoshi/Documents/GitHub/Water-Level-Prediction/Data/test_data/Coastal Bend Met Measurements_2025-1.xlsx')

datetime_col_water = [c for c in water_2025_raw.columns if 'date' in c.lower() or 'time' in c.lower()][0]
datetime_col_met = [c for c in met_2025_raw.columns if 'date' in c.lower() or 'time' in c.lower()][0]

water_2025_raw['DateTime'] = pd.to_datetime(water_2025_raw[datetime_col_water], errors='coerce')
water_2025 = water_2025_raw[water_2025_raw['DateTime'].notna()].copy()
water_2025 = water_2025[water_2025['DateTime'] >= '2025-01-01'].copy()

met_2025_raw['DateTime'] = pd.to_datetime(met_2025_raw[datetime_col_met], errors='coerce')
met_2025 = met_2025_raw[met_2025_raw['DateTime'].notna()].copy()
met_2025 = met_2025[met_2025['DateTime'] >= '2025-01-01'].copy()

TRAINING_STATIONS = ['005', '008', '013', '202']

water_cols = ['DateTime'] + [c for c in water_2025.columns if any(c.startswith(f'{s}-') for s in TRAINING_STATIONS) and 'pwl' in c]
met_cols = ['DateTime'] + [c for c in met_2025.columns if any(c.startswith(f'{s}-') for s in TRAINING_STATIONS)]

water_2025 = water_2025[water_cols].copy()
met_2025 = met_2025[met_cols].copy()

test_2025 = water_2025.rename(columns={'DateTime': 'datetime'})
met_2025 = met_2025.rename(columns={'DateTime': 'datetime'})
test_2025 = test_2025.merge(met_2025, on='datetime', how='left')
test_2025 = test_2025.loc[:, ~test_2025.columns.duplicated()]

# HYBRID FILLING FOR BASE DATA
for col in test_2025.columns:
    if col != 'datetime' and test_2025[col].dtype in [np.float64, np.float32]:
        if test_2025[col].isna().any():
            test_2025[col] = test_2025[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

print(f"  Loaded: {test_2025.shape}")

# FEATURE ENGINEERING

print("\n[2/5] Creating Features...")

TARGET = '005-pwl'

# Tidal constituents
M2, S2, N2, K1 = 12.4206, 12.0000, 12.6583, 23.9345
test_2025['hours_since_epoch'] = (test_2025['datetime'] - pd.Timestamp('2000-01-01')).dt.total_seconds() / 3600

test_2025['tide_M2_sin'] = np.sin(2 * np.pi * test_2025['hours_since_epoch'] / M2)
test_2025['tide_M2_cos'] = np.cos(2 * np.pi * test_2025['hours_since_epoch'] / M2)
test_2025['tide_S2_sin'] = np.sin(2 * np.pi * test_2025['hours_since_epoch'] / S2)
test_2025['tide_S2_cos'] = np.cos(2 * np.pi * test_2025['hours_since_epoch'] / S2)
test_2025['tide_N2_sin'] = np.sin(2 * np.pi * test_2025['hours_since_epoch'] / N2)
test_2025['tide_N2_cos'] = np.cos(2 * np.pi * test_2025['hours_since_epoch'] / N2)
test_2025['tide_K1_sin'] = np.sin(2 * np.pi * test_2025['hours_since_epoch'] / K1)
test_2025['tide_K1_cos'] = np.cos(2 * np.pi * test_2025['hours_since_epoch'] / K1)

lunar_period = 29.53 * 24
test_2025['lunar_phase'] = (test_2025['hours_since_epoch'] % lunar_period) / lunar_period
test_2025['spring_neap'] = np.cos(2 * np.pi * test_2025['lunar_phase'])

for station in ['005', '013', '202']:
    wsd_col = f'{station}-wsd'
    wdr_col = f'{station}-wdr'
    if wsd_col in test_2025.columns and wdr_col in test_2025.columns:
        test_2025[f'{station}_wind_u'] = -test_2025[wsd_col] * np.sin(np.radians(test_2025[wdr_col]))
        test_2025[f'{station}_wind_v'] = -test_2025[wsd_col] * np.cos(np.radians(test_2025[wdr_col]))
        test_2025[f'{station}_wind_stress'] = test_2025[wsd_col] ** 2 / 100

if '005-bpr' in test_2025.columns:
    test_2025['pressure_change_1h'] = test_2025['005-bpr'].diff(1).fillna(0)
    test_2025['pressure_change_3h'] = test_2025['005-bpr'].diff(3).fillna(0)
    test_2025['pressure_change_6h'] = test_2025['005-bpr'].diff(6).fillna(0)
    test_2025['pressure_accel'] = test_2025['pressure_change_1h'].diff(1).fillna(0)

if '013-bpr' in test_2025.columns and '202-bpr' in test_2025.columns:
    test_2025['pressure_gradient_NS'] = test_2025['013-bpr'] - test_2025['202-bpr']
    test_2025['pressure_gradient_bay'] = test_2025['005-bpr'] - test_2025['202-bpr']

if '005-wtp' in test_2025.columns:
    test_2025['temp_gradient_005'] = test_2025['005-wtp'].diff(1).fillna(0)
if '013-wtp' in test_2025.columns and '005-wtp' in test_2025.columns:
    test_2025['water_temp_diff_013'] = test_2025['013-wtp'] - test_2025['005-wtp']
if '202-wtp' in test_2025.columns and '005-wtp' in test_2025.columns:
    test_2025['water_temp_diff_202'] = test_2025['202-wtp'] - test_2025['005-wtp']

if '013-pwl' in test_2025.columns and '202-pwl' in test_2025.columns:
    test_2025['gradient_NS'] = test_2025['013-pwl'] - test_2025['202-pwl']
    test_2025['gradient_NS_rate'] = test_2025['gradient_NS'].diff(1).fillna(0)
if '005-pwl' in test_2025.columns and '008-pwl' in test_2025.columns:
    test_2025['gradient_bay_gulf'] = test_2025['005-pwl'] - test_2025['008-pwl']

pwl_cols = [c for c in test_2025.columns if c in ['005-pwl', '008-pwl', '013-pwl', '202-pwl']]
if len(pwl_cols) >= 2:
    test_2025['spatial_mean'] = test_2025[pwl_cols].mean(axis=1)
    test_2025['spatial_std'] = test_2025[pwl_cols].std(axis=1).fillna(0)
    test_2025['packery_anomaly'] = test_2025['005-pwl'] - test_2025['spatial_mean']

test_2025['packery_lag_1h'] = test_2025['005-pwl'].shift(1)
test_2025['packery_lag_6h'] = test_2025['005-pwl'].shift(6)
test_2025['packery_lag_12h'] = test_2025['005-pwl'].shift(12)
test_2025['packery_lag_24h'] = test_2025['005-pwl'].shift(24)
test_2025['packery_lag_48h'] = test_2025['005-pwl'].shift(48)

test_2025['packery_std_6h'] = test_2025['005-pwl'].rolling(6, min_periods=1).std().fillna(0)
test_2025['packery_std_12h'] = test_2025['005-pwl'].rolling(12, min_periods=1).std().fillna(0)
test_2025['packery_mean_24h'] = test_2025['005-pwl'].rolling(24, min_periods=1).mean()
test_2025['packery_std_24h'] = test_2025['005-pwl'].rolling(24, min_periods=1).std().fillna(0)

test_2025['packery_rate_1h'] = test_2025['005-pwl'].diff(1).fillna(0)
test_2025['packery_rate_6h'] = test_2025['005-pwl'].diff(6).fillna(0)

test_2025['hour'] = test_2025['datetime'].dt.hour
test_2025['day_of_week'] = test_2025['datetime'].dt.dayofweek
test_2025['hour_sin'] = np.sin(2 * np.pi * test_2025['hour'] / 24)
test_2025['hour_cos'] = np.cos(2 * np.pi * test_2025['hour'] / 24)
test_2025['day_sin'] = np.sin(2 * np.pi * test_2025['day_of_week'] / 7)
test_2025['day_cos'] = np.cos(2 * np.pi * test_2025['day_of_week'] / 7)

features_df = pd.read_csv('../Data/modeling_enhanced/features.csv')
all_features = features_df['feature'].tolist()

print(f"  Created 48 features")

# CORRECT MODEL ARCHITECTURES

class LSTMModel_Correct(nn.Module):
    """LSTM with hidden_size=64 and intermediate fc1 layer"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.relu(self.fc1(lstm_out[:, -1, :]))
        return self.fc2(x)

def create_sequences(data, features, target, sequence_length=24):
    X_seq, y_seq, datetimes = [], [], []
    for i in range(sequence_length, len(data)):
        X_seq.append(data[features].iloc[i-sequence_length:i].values)
        y_seq.append(data[target].iloc[i])
        datetimes.append(data['datetime'].iloc[i])
    return np.array(X_seq), np.array(y_seq), datetimes

# TEST ALL SCENARIOS

scenarios = {
    'Scenario1_1h': {'horizon': 1, 'n_features': 48, 'sequence_length': 24},
    'Scenario2_6h': {'horizon': 6, 'n_features': 45, 'sequence_length': 24},
    'Scenario3_12h': {'horizon': 12, 'n_features': 42, 'sequence_length': 24}
}

all_results = []

for scenario_name, config in scenarios.items():
    print("\n" + "="*80)
    print(f"[3/5] TESTING: {scenario_name}")
    print("="*80)
    
    horizon = config['horizon']
    n_features = config['n_features']
    seq_len = config['sequence_length']
    
    test_2025['target'] = test_2025[TARGET].shift(-horizon)
    test_clean = test_2025.dropna(subset=['target', TARGET]).reset_index(drop=True)
    
    scenario_features = all_features[:n_features]
    
    print(f"  Samples: {len(test_clean):,}, Features: {n_features}")
    
    datetime_test = test_clean['datetime'].values
    y_test = test_clean['target'].values
    persistence = test_clean[TARGET].values
    
    pers_rmse = np.sqrt(mean_squared_error(y_test, persistence))
    print(f"  Persistence: {pers_rmse*100:.2f} cm")
    
    results = {'Scenario': scenario_name, 'Horizon': horizon, 'Samples': len(y_test), 
               'Persistence_RMSE': pers_rmse*100}
    

    # MODEL 1: XGBoost

    
    print(f"\n  [1/4] XGBoost...", end=" ")
    try:
        with open(f'models/{scenario_name}/xgboost.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        X_test = test_clean[scenario_features].values
        xgb_pred = xgb_model.predict(X_test)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_r2 = r2_score(y_test, xgb_pred)
        xgb_within15 = np.mean(np.abs(y_test - xgb_pred) <= 0.15) * 100
        xgb_imp = (1 - xgb_rmse/pers_rmse) * 100
        print(f"✅ {xgb_rmse*100:.2f}cm (R²={xgb_r2:.3f}, {xgb_within15:.0f}% in 15cm, {xgb_imp:+.1f}%)")
        results.update({'XGBoost_RMSE': xgb_rmse*100, 'XGBoost_R2': xgb_r2, 'XGBoost_Within15': xgb_within15, 'XGBoost_Imp': xgb_imp})
    except Exception as e:
        print(f"❌ {e}")
        xgb_pred = np.zeros_like(y_test)
        results.update({'XGBoost_RMSE': np.nan, 'XGBoost_R2': np.nan, 'XGBoost_Within15': np.nan, 'XGBoost_Imp': np.nan})
    

    # MODEL 2: Random Forest

    
    print(f"  [2/4] Random Forest...", end=" ")
    try:
        with open(f'models/{scenario_name}/random_forest.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        rf_pred = rf_model.predict(X_test)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_r2 = r2_score(y_test, rf_pred)
        rf_within15 = np.mean(np.abs(y_test - rf_pred) <= 0.15) * 100
        rf_imp = (1 - rf_rmse/pers_rmse) * 100
        print(f"✅ {rf_rmse*100:.2f}cm (R²={rf_r2:.3f}, {rf_within15:.0f}% in 15cm, {rf_imp:+.1f}%)")
        results.update({'RF_RMSE': rf_rmse*100, 'RF_R2': rf_r2, 'RF_Within15': rf_within15, 'RF_Imp': rf_imp})
    except Exception as e:
        print(f"❌ {e}")
        rf_pred = np.zeros_like(y_test)
        results.update({'RF_RMSE': np.nan, 'RF_R2': np.nan, 'RF_Within15': np.nan, 'RF_Imp': np.nan})
    

# MODEL 3: LSTM - WITH INVERSE TRANSFORM FIX

    print(f"  [3/4] LSTM...", end=" ")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        checkpoint = torch.load(f'models/{scenario_name}/lstm.pth', 
                            map_location=device, 
                            weights_only=False)
        
        # Create correct architecture
        lstm_model = LSTMModel_Correct(input_size=n_features, hidden_size=64, num_layers=2, dropout=0.2)
        lstm_model.load_state_dict(checkpoint['model_state_dict'])
        lstm_model.to(device)
        lstm_model.eval()
        
        # HYBRID FILLING
        print(f"(hybrid-fill...", end=" ")
        
        test_lstm = test_clean.copy()
        
        for col in scenario_features:
            if col in test_lstm.columns:
                if test_lstm[col].isna().any():
                    test_lstm[col] = test_lstm[col].fillna(method='ffill')
                    test_lstm[col] = test_lstm[col].fillna(method='bfill')
                    test_lstm[col] = test_lstm[col].fillna(0)
        
        nan_count = test_lstm[scenario_features].isna().sum().sum()
        if nan_count > 0:
            test_lstm[scenario_features] = test_lstm[scenario_features].fillna(0)
        
        print(f"✓)", end=" ")
        
        # Create sequences
        X_seq, y_seq, dt_seq = create_sequences(test_lstm, scenario_features, 'target', seq_len)
        
        if np.isnan(X_seq).any():
            X_seq = np.nan_to_num(X_seq, nan=0.0)
        
        # Load BOTH scalers
        scaler_X = checkpoint['scaler_X']
        scaler_y = checkpoint['scaler_y']  # CRITICAL!
        
        print(f"(scale+predict...", end=" ")
        
        # Scale features
        X_seq_scaled = np.array([scaler_X.transform(seq) for seq in X_seq])
        
        if np.isnan(X_seq_scaled).any():
            X_seq_scaled = np.nan_to_num(X_seq_scaled, nan=0.0)
        
        # Predict (returns SCALED predictions)
        X_tensor = torch.FloatTensor(X_seq_scaled).to(device)
        
        with torch.no_grad():
            lstm_pred_scaled = lstm_model(X_tensor).cpu().numpy()
        
        # CRITICAL FIX: Inverse transform to original space
        print(f"inverse)", end=" ")
        lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled).flatten()
        
        # Metrics (now in unscaled space)
        y_test_lstm = y_seq
        persistence_lstm = test_lstm[TARGET].iloc[seq_len:].values
        
        lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm, lstm_pred))
        lstm_r2 = r2_score(y_test_lstm, lstm_pred)
        lstm_within15 = np.mean(np.abs(y_test_lstm - lstm_pred) <= 0.15) * 100
        pers_rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, persistence_lstm))
        lstm_imp = (1 - lstm_rmse/pers_rmse_lstm) * 100
        
        print(f"✅ {lstm_rmse*100:.2f}cm (R²={lstm_r2:.3f}, {lstm_within15:.0f}% in 15cm, {lstm_imp:+.1f}%)")
        results.update({'LSTM_RMSE': lstm_rmse*100, 'LSTM_R2': lstm_r2, 'LSTM_Within15': lstm_within15, 'LSTM_Imp': lstm_imp})
        
    except Exception as e:
        print(f"❌ {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        results.update({'LSTM_RMSE': np.nan, 'LSTM_R2': np.nan, 'LSTM_Within15': np.nan, 'LSTM_Imp': np.nan})
    

    # MODEL 4: Ensemble

    
    print(f"  [4/4] Ensemble...", end=" ")
    try:
        valid_preds = []
        if not np.isnan(results.get('XGBoost_RMSE', np.nan)):
            valid_preds.append(xgb_pred)
        if not np.isnan(results.get('RF_RMSE', np.nan)):
            valid_preds.append(rf_pred)
        
        ensemble_pred = np.mean(valid_preds, axis=0)
        ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ens_r2 = r2_score(y_test, ensemble_pred)
        ens_within15 = np.mean(np.abs(y_test - ensemble_pred) <= 0.15) * 100
        ens_imp = (1 - ens_rmse/pers_rmse) * 100
        
        print(f"✅ {ens_rmse*100:.2f}cm (R²={ens_r2:.3f}, {ens_within15:.0f}% in 15cm, {ens_imp:+.1f}%)")
        results.update({'Ensemble_RMSE': ens_rmse*100, 'Ensemble_R2': ens_r2, 'Ensemble_Within15': ens_within15, 'Ensemble_Imp': ens_imp})
        
    except Exception as e:
        print(f"❌ {e}")
        results.update({'Ensemble_RMSE': np.nan, 'Ensemble_R2': np.nan, 'Ensemble_Within15': np.nan, 'Ensemble_Imp': np.nan})
    
    # Note about GNN
    print(f"  ℹ️  GNN: Requires PyTorch Geometric (GAT layers) - validation: 6.33cm @ 6h")
    results.update({'GNN_RMSE': np.nan, 'GNN_R2': np.nan, 'GNN_Within15': np.nan, 'GNN_Imp': np.nan})
    
    all_results.append(results)
    
    all_results.append(results)
    

    # SAVE PREDICTIONS WITH PROPER ALIGNMENT

    
    print(f"  💾 Saving predictions...", end=" ")
    try:
        # For XGBoost, RF, Ensemble: predictions match y_test exactly
        xgb_pred_save = xgb_pred if 'xgb_pred' in locals() else np.full_like(y_test, np.nan)
        rf_pred_save = rf_pred if 'rf_pred' in locals() else np.full_like(y_test, np.nan)
        ensemble_pred_save = ensemble_pred if 'ensemble_pred' in locals() else np.full_like(y_test, np.nan)
        
        # For LSTM: predictions are shorter by seq_len
        # Create full-length array with NaN for first seq_len entries
        lstm_pred_save = np.full(len(y_test), np.nan)
        if 'lstm_pred' in locals() and lstm_pred is not None:
            # LSTM predictions start from index seq_len
            lstm_pred_save[seq_len:] = lstm_pred
        
        pred_df = pd.DataFrame({
            'datetime': datetime_test,
            'actual': y_test,
            'persistence': persistence,
            'xgboost': xgb_pred_save,
            'random_forest': rf_pred_save,
            'lstm': lstm_pred_save,  # ✅ LSTM PREDICTIONS NOW INCLUDED
            'ensemble': ensemble_pred_save
        })
        
        pred_df.to_csv(f'models/{scenario_name}/predictions_2025_ALL_MODELS_FINAL.csv', index=False)
        
        # Report what was saved
        n_lstm = (~np.isnan(pred_df['lstm'])).sum()
        print(f"✅ Saved {len(pred_df)} samples (LSTM: {n_lstm} valid predictions)")
        
    except Exception as e:
        print(f"❌ Save error: {e}")
        import traceback
        traceback.print_exc()

# SUMMARY

print("\n" + "="*80)
print("[4/5] FINAL RESULTS - ALL MODELS ON 2025 TEST")
print("="*80)

results_df = pd.DataFrame(all_results)

for _, row in results_df.iterrows():
    print(f"\n{row['Scenario']} ({int(row['Horizon'])}h ahead, {int(row['Samples']):,} samples):")
    print(f"  {'Model':<20} {'RMSE':<10} {'R²':<10} {'Within 15cm':<12} {'vs Pers'}")
    print(f"  {'-'*65}")
    print(f"  {'Persistence':<20} {row['Persistence_RMSE']:>8.2f}cm")
    
    for model in ['XGBoost', 'RF', 'LSTM', 'Ensemble']:
        rmse = row.get(f'{model}_RMSE', np.nan)
        r2 = row.get(f'{model}_R2', np.nan)
        within = row.get(f'{model}_Within15', np.nan)
        imp = row.get(f'{model}_Imp', np.nan)
        
        if not np.isnan(rmse):
            print(f"  {model:<20} {rmse:>8.2f}cm {r2:>8.3f}  {within:>10.0f}%  {imp:>+7.1f}%")

print("\n" + "="*80)
print("[5/5] BEST MODELS PER SCENARIO")
print("="*80)

for _, row in results_df.iterrows():
    model_results = {}
    for model in ['XGBoost', 'RF', 'LSTM', 'Ensemble']:
        rmse = row.get(f'{model}_RMSE', np.nan)
        if not np.isnan(rmse):
            model_results[model] = rmse
    
    if model_results:
        best_model = min(model_results, key=model_results.get)
        best_rmse = model_results[best_model]
        print(f"\n{row['Scenario']}: 🏆 {best_model} = {best_rmse:.2f} cm")

results_df.to_csv('models/2025_ALL_MODELS_comparison_FINAL.csv', index=False)
print(f"\n💾 Saved: models/2025_ALL_MODELS_comparison_FINAL.csv")

print("\n" + "="*80)
print("✅ COMPREHENSIVE TEST COMPLETE!")
print("="*80)
