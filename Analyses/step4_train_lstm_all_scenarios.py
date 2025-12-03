import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 4: LSTM FOR ALL SCENARIOS (1h, 6h, 12h)")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[Device]: {device}")

train = pd.read_csv('../Data/modeling_enhanced/train_data.csv')
val = pd.read_csv('../Data/modeling_enhanced/val_data.csv')
features_df = pd.read_csv('../Data/modeling_enhanced/features.csv')

all_features = features_df['feature'].tolist()
TARGET = '005-pwl'

class WaterLevelLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(WaterLevelLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, sequence_length=24):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        X = self.features[idx:idx+self.sequence_length]
        y = self.targets[idx+self.sequence_length]
        return torch.FloatTensor(X), torch.FloatTensor([y])

def train_lstm(model, train_loader, val_loader, epochs=50, lr=0.0001, patience=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = model.state_dict().copy()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                continue
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            if torch.isnan(outputs).any():
                print(f"  WARNING: NaN in outputs at epoch {epoch+1}, skipping batch")
                continue
            
            loss = criterion(outputs, y_batch)
            
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at epoch {epoch+1}, skipping batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        if train_batches == 0:
            print(f"  ERROR: All batches skipped at epoch {epoch+1}")
            break
        
        train_loss /= train_batches
        
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                    continue
                
                outputs = model(X_batch)
                
                if torch.isnan(outputs).any():
                    continue
                
                loss = criterion(outputs, y_batch)
                
                if torch.isnan(loss):
                    continue
                
                val_loss += loss.item()
                val_batches += 1
        
        if val_batches == 0:
            print(f"  ERROR: All validation batches skipped at epoch {epoch+1}")
            break
        
        val_loss /= val_batches
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_model_state)
    return model

def predict_lstm(model, dataloader):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            
            if torch.isnan(X_batch).any():
                continue
            
            outputs = model(X_batch)
            
            if torch.isnan(outputs).any():
                continue
            
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions).flatten()

scenarios = {
    'Scenario1_1h': {
        'horizon': 1,
        'excluded_lags': [],
        'sequence_length': 24
    },
    'Scenario2_6h': {
        'horizon': 6,
        'excluded_lags': ['packery_lag_1h', 'packery_lag_2h', 'packery_lag_3h',
                          'packery_mean_6h', 'packery_std_6h', 'packery_rate_1h'],
        'sequence_length': 48
    },
    'Scenario3_12h': {
        'horizon': 12,
        'excluded_lags': ['packery_lag_1h', 'packery_lag_2h', 'packery_lag_3h', 
                          'packery_lag_6h', 'packery_mean_6h', 'packery_std_6h',
                          'packery_mean_12h', 'packery_std_12h',
                          'packery_rate_1h', 'packery_rate_6h'],
        'sequence_length': 72
    }
}

all_results = []

for scenario_name, config in scenarios.items():
    print("\n" + "="*80)
    print(f"{scenario_name} - LSTM {config['horizon']}-HOUR AHEAD")
    print("="*80)
    
    horizon = config['horizon']
    excluded = config['excluded_lags']
    seq_length = config['sequence_length']
    
    scenario_features = [f for f in all_features if f not in excluded]
    
    print(f"\n  Features: {len(scenario_features)}")
    print(f"  Sequence length: {seq_length}")
    
    train['target'] = train[TARGET].shift(-horizon)
    val['target'] = val[TARGET].shift(-horizon)
    
    train_clean = train.dropna(subset=['target'])
    val_clean = val.dropna(subset=['target'])
    
    for col in scenario_features:
        if train_clean[col].isna().any():
            train_clean[col] = train_clean[col].fillna(train_clean[col].mean())
        if val_clean[col].isna().any():
            val_clean[col] = val_clean[col].fillna(val_clean[col].mean())
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(train_clean[scenario_features])
    y_train = scaler_y.fit_transform(train_clean['target'].values.reshape(-1, 1)).flatten()
    
    X_val = scaler_X.transform(val_clean[scenario_features])
    y_val = scaler_y.transform(val_clean['target'].values.reshape(-1, 1)).flatten()
    
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("  ERROR: NaN/Inf in training features after scaling")
        continue
    
    if np.isnan(y_train).any() or np.isinf(y_train).any():
        print("  ERROR: NaN/Inf in training targets after scaling")
        continue
    
    train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length=seq_length)
    val_dataset = TimeSeriesDataset(X_val, y_val, sequence_length=seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"  Training sequences: {len(train_dataset):,}")
    print(f"  Validation sequences: {len(val_dataset):,}")
    
    print(f"\n[Training LSTM]")
    
    model = WaterLevelLSTM(
        input_size=len(scenario_features),
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    model = train_lstm(model, train_loader, val_loader, epochs=100, lr=0.0001, patience=15)
    
    print(f"\n[Generating Predictions]")
    
    lstm_pred_scaled = predict_lstm(model, val_loader)
    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
    
    actual_targets = val_clean['target'].values[seq_length:]
    
    if len(lstm_pred) < len(actual_targets):
        actual_targets = actual_targets[:len(lstm_pred)]
    
    lstm_residuals = actual_targets - lstm_pred
    
    lstm_mse = mean_squared_error(actual_targets, lstm_pred)
    lstm_rmse = np.sqrt(lstm_mse)
    lstm_mae = mean_absolute_error(actual_targets, lstm_pred)
    lstm_r2 = r2_score(actual_targets, lstm_pred)
    lstm_medae = median_absolute_error(actual_targets, lstm_pred)
    lstm_maxerr = np.max(np.abs(lstm_residuals))
    lstm_residual_std = np.std(lstm_residuals)
    lstm_within15 = np.mean(np.abs(lstm_residuals) <= 0.15) * 100
    
    print(f"  RMSE: {lstm_rmse:.4f} m ({lstm_rmse*100:.2f} cm)")
    print(f"  R²: {lstm_r2:.4f}")
    print(f"  Within 15cm: {lstm_within15:.1f}%")
    
    val_pers = val_clean.iloc[seq_length:seq_length+len(lstm_pred)].copy()
    persistence_pred = val_pers[TARGET].values
    
    pers_residuals = actual_targets - persistence_pred
    
    pers_mse = mean_squared_error(actual_targets, persistence_pred)
    pers_rmse = np.sqrt(pers_mse)
    pers_mae = mean_absolute_error(actual_targets, persistence_pred)
    pers_r2 = r2_score(actual_targets, persistence_pred)
    pers_medae = median_absolute_error(actual_targets, persistence_pred)
    pers_maxerr = np.max(np.abs(pers_residuals))
    pers_residual_std = np.std(pers_residuals)
    pers_within15 = np.mean(np.abs(pers_residuals) <= 0.15) * 100
    
    improvement = (1 - lstm_rmse/pers_rmse) * 100
    print(f"  Persistence RMSE: {pers_rmse:.4f} m ({pers_rmse*100:.2f} cm)")
    print(f"  LSTM Improvement: {improvement:+.1f}%")
    
    import os
    os.makedirs(f'models/{scenario_name}', exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'features': scenario_features,
        'sequence_length': seq_length
    }, f'models/{scenario_name}/lstm.pth')
    
    pred_df = pd.DataFrame({
        'datetime': val_pers['datetime'].values,
        'actual': actual_targets,
        'persistence': persistence_pred,
        'lstm': lstm_pred
    })
    pred_df.to_csv(f'models/{scenario_name}/lstm_predictions.csv', index=False)
    
    scenario_results = {
        'Scenario': [scenario_name, scenario_name],
        'Model': ['Persistence', 'LSTM'],
        'Horizon_Hours': [horizon, horizon],
        'Central Frequency 15cm': [pers_within15, lstm_within15],
        'MSE': [pers_mse, lstm_mse],
        'RMSE': [pers_rmse, lstm_rmse],
        'R2': [pers_r2, lstm_r2],
        'MAE': [pers_mae, lstm_mae],
        'MedAE': [pers_medae, lstm_medae],
        'MaxErr': [pers_maxerr, lstm_maxerr],
        'residual Stdev': [pers_residual_std, lstm_residual_std]
    }
    
    scenario_df = pd.DataFrame(scenario_results)
    scenario_df.to_csv(f'models/{scenario_name}/lstm_results.csv', index=False)
    
    all_results.append(scenario_df)
    
    print(f"\n  ✅ Saved: models/{scenario_name}/lstm.pth")

print("\n" + "="*80)
print("LSTM RESULTS - ALL SCENARIOS")
print("="*80)

combined_df = pd.concat(all_results, ignore_index=True)
print("\n" + combined_df.to_string(index=False))

combined_df.to_csv('models/lstm_all_scenarios_results.csv', index=False)
print(f"\n✅ Saved: models/lstm_all_scenarios_results.csv")

print("\n" + "="*80)
print("✅ STEP 4 COMPLETE - LSTM TRAINED FOR ALL SCENARIOS!")
