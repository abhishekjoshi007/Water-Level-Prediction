import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader as GeoDataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 5: GNN FOR ALL SCENARIOS (1h, 6h, 12h) - FIXED")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[Device]: {device}")

train = pd.read_csv('../Data/modeling_enhanced/train_data.csv')
val = pd.read_csv('../Data/modeling_enhanced/val_data.csv')
features_df = pd.read_csv('../Data/modeling_enhanced/features.csv')

all_features = features_df['feature'].tolist()
TARGET = '005-pwl'

STATIONS = ['005', '008', '013', '202']
STATION_TO_IDX = {s: i for i, s in enumerate(STATIONS)}
TARGET_IDX = STATION_TO_IDX['005']

# Bidirectional edges: 005-008, 005-013, 005-202, 013-202
edge_index = torch.tensor([
    [0, 1, 0, 2, 0, 3, 2, 3],
    [1, 0, 2, 0, 3, 0, 3, 2]
], dtype=torch.long)

class WaterLevelGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim=64, num_heads=4):
        super(WaterLevelGNN, self).__init__()
        
        self.gat1 = GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.2)
        self.gat3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=0.2)
        
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)
        x = self.gat3(x, edge_index)
        
        # Extract target node (005) from each graph
        # batch tells us which graph each node belongs to
        num_graphs = batch.max().item() + 1
        target_features = []
        for i in range(num_graphs):
            graph_mask = (batch == i)
            graph_nodes = x[graph_mask]
            target_features.append(graph_nodes[TARGET_IDX])
        
        target_features = torch.stack(target_features)
        
        out = F.relu(self.fc1(target_features))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def create_graph_features(df, station_features_list, global_features_list):
    """Extract station-specific and global features"""
    graphs = []
    
    for idx in range(len(df)):
        node_features = []
        
        # Create features for each station node
        for station in STATIONS:
            station_feats = []
            
            # Station-specific features (water level, met data)
            for feat_name in station_features_list:
                col_name = f'{station}-{feat_name}'
                if col_name in df.columns:
                    val = df.iloc[idx][col_name]
                    station_feats.append(val if not np.isnan(val) else 0.0)
                else:
                    station_feats.append(0.0)  # Missing sensor
            
            # Add global features (time, tides, etc.) to all nodes
            for feat in global_features_list:
                if feat in df.columns:
                    val = df.iloc[idx][feat]
                    station_feats.append(val if not np.isnan(val) else 0.0)
                else:
                    station_feats.append(0.0)
            
            node_features.append(station_feats)
        
        graphs.append(node_features)
    
    return np.array(graphs)

def train_gnn(model, train_loader, val_loader, epochs=100, lr=0.0001, patience=15):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = model.state_dict().copy()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_batches = 0
        
        for data in train_loader:
            data = data.to(device)
            
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data.x, data.edge_index, data.batch)
                loss = criterion(output, data.y)
                val_loss += loss.item()
                val_batches += 1
        
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

def predict_gnn(model, dataloader):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            predictions.extend(output.cpu().numpy().flatten())
    
    return np.array(predictions)

scenarios = {
    'Scenario1_1h': {
        'horizon': 1,
        'excluded_lags': []
    },
    'Scenario2_6h': {
        'horizon': 6,
        'excluded_lags': ['packery_lag_1h', 'packery_lag_2h', 'packery_lag_3h',
                          'packery_mean_6h', 'packery_std_6h', 'packery_rate_1h']
    },
    'Scenario3_12h': {
        'horizon': 12,
        'excluded_lags': ['packery_lag_1h', 'packery_lag_2h', 'packery_lag_3h', 
                          'packery_lag_6h', 'packery_mean_6h', 'packery_std_6h',
                          'packery_mean_12h', 'packery_std_12h',
                          'packery_rate_1h', 'packery_rate_6h']
    }
}

station_features_base = ['pwl', 'atp', 'wtp', 'wsd', 'wgt', 'wdr', 'bpr']

all_results = []

for scenario_name, config in scenarios.items():
    print("\n" + "="*80)
    print(f"{scenario_name} - GNN {config['horizon']}-HOUR AHEAD")
    print("="*80)
    
    horizon = config['horizon']
    excluded = config['excluded_lags']
    
    # Global features (non-station specific)
    global_features = [f for f in all_features if f not in excluded and 
                       not any(f.startswith(f'{s}-') for s in STATIONS)]
    
    print(f"\n  Station features per node: {len(station_features_base)}")
    print(f"  Global features per node: {len(global_features)}")
    print(f"  Total features per node: {len(station_features_base) + len(global_features)}")
    
    # Create target column
    train_copy = train.copy()
    val_copy = val.copy()
    
    train_copy['target'] = train_copy[TARGET].shift(-horizon)
    val_copy['target'] = val_copy[TARGET].shift(-horizon)
    
    # Drop NaN targets
    train_clean = train_copy.dropna(subset=['target']).reset_index(drop=True)
    val_clean = val_copy.dropna(subset=['target']).reset_index(drop=True)
    
    # Fill any remaining NaN in features
    for col in train_clean.columns:
        if train_clean[col].dtype in [np.float64, np.float32]:
            if train_clean[col].isna().any():
                train_clean[col] = train_clean[col].fillna(train_clean[col].mean())
            if val_clean[col].isna().any():
                val_clean[col] = val_clean[col].fillna(val_clean[col].mean())
    
    # Create graph features BEFORE scaling
    print(f"\n[Creating Graph Features]")
    train_graphs_array = create_graph_features(train_clean, station_features_base, global_features)
    val_graphs_array = create_graph_features(val_clean, station_features_base, global_features)
    
    print(f"  Training samples: {len(train_graphs_array):,}")
    print(f"  Validation samples: {len(val_graphs_array):,}")
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Flatten for scaling, then reshape
    n_train = train_graphs_array.shape[0]
    n_val = val_graphs_array.shape[0]
    n_nodes = train_graphs_array.shape[1]
    n_features = train_graphs_array.shape[2]
    
    train_flat = train_graphs_array.reshape(-1, n_features)
    val_flat = val_graphs_array.reshape(-1, n_features)
    
    train_flat_scaled = scaler_X.fit_transform(train_flat)
    val_flat_scaled = scaler_X.transform(val_flat)
    
    train_graphs_scaled = train_flat_scaled.reshape(n_train, n_nodes, n_features)
    val_graphs_scaled = val_flat_scaled.reshape(n_val, n_nodes, n_features)
    
    # Scale targets
    y_train = scaler_y.fit_transform(train_clean['target'].values.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(val_clean['target'].values.reshape(-1, 1)).flatten()
    
    # Create PyG Data objects
    print(f"\n[Creating PyG Graphs]")
    train_pyg_graphs = []
    for i in range(len(train_graphs_scaled)):
        x = torch.FloatTensor(train_graphs_scaled[i])
        y = torch.FloatTensor([y_train[i]])
        data = Data(x=x, edge_index=edge_index, y=y)
        train_pyg_graphs.append(data)
    
    val_pyg_graphs = []
    for i in range(len(val_graphs_scaled)):
        x = torch.FloatTensor(val_graphs_scaled[i])
        y = torch.FloatTensor([y_val[i]])
        data = Data(x=x, edge_index=edge_index, y=y)
        val_pyg_graphs.append(data)
    
    train_loader = GeoDataLoader(train_pyg_graphs, batch_size=64, shuffle=False)
    val_loader = GeoDataLoader(val_pyg_graphs, batch_size=64, shuffle=False)
    
    num_node_features = len(station_features_base) + len(global_features)
    
    print(f"\n[Training GNN]")
    
    model = WaterLevelGNN(
        num_node_features=num_node_features,
        hidden_dim=64,
        num_heads=4
    ).to(device)
    
    model = train_gnn(model, train_loader, val_loader, epochs=100, lr=0.0001, patience=15)
    
    print(f"\n[Generating Predictions]")
    
    gnn_pred_scaled = predict_gnn(model, val_loader)
    gnn_pred = scaler_y.inverse_transform(gnn_pred_scaled.reshape(-1, 1)).flatten()
    
    # CRITICAL FIX: Use UNSCALED original data for targets and persistence
    actual_targets = val_clean['target'].values
    persistence_pred = val_clean[TARGET].values
    
    # Calculate metrics
    gnn_residuals = actual_targets - gnn_pred
    gnn_mse = mean_squared_error(actual_targets, gnn_pred)
    gnn_rmse = np.sqrt(gnn_mse)
    gnn_mae = mean_absolute_error(actual_targets, gnn_pred)
    gnn_r2 = r2_score(actual_targets, gnn_pred)
    gnn_medae = median_absolute_error(actual_targets, gnn_pred)
    gnn_maxerr = np.max(np.abs(gnn_residuals))
    gnn_residual_std = np.std(gnn_residuals)
    gnn_within15 = np.mean(np.abs(gnn_residuals) <= 0.15) * 100
    
    print(f"  RMSE: {gnn_rmse:.4f} m ({gnn_rmse*100:.2f} cm)")
    print(f"  R²: {gnn_r2:.4f}")
    print(f"  Within 15cm: {gnn_within15:.1f}%")
    
    # Persistence baseline (UNSCALED)
    pers_residuals = actual_targets - persistence_pred
    pers_mse = mean_squared_error(actual_targets, persistence_pred)
    pers_rmse = np.sqrt(pers_mse)
    pers_mae = mean_absolute_error(actual_targets, persistence_pred)
    pers_r2 = r2_score(actual_targets, persistence_pred)
    pers_medae = median_absolute_error(actual_targets, persistence_pred)
    pers_maxerr = np.max(np.abs(pers_residuals))
    pers_residual_std = np.std(pers_residuals)
    pers_within15 = np.mean(np.abs(pers_residuals) <= 0.15) * 100
    
    improvement = (1 - gnn_rmse/pers_rmse) * 100
    print(f"  Persistence RMSE: {pers_rmse:.4f} m ({pers_rmse*100:.2f} cm)")
    print(f"  GNN Improvement: {improvement:+.1f}%")
    
    # Save
    import os
    os.makedirs(f'models/{scenario_name}', exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'station_features': station_features_base,
        'global_features': global_features
    }, f'models/{scenario_name}/gnn_fixed.pth')
    
    pred_df = pd.DataFrame({
        'datetime': val_clean['datetime'].values,
        'actual': actual_targets,
        'persistence': persistence_pred,
        'gnn': gnn_pred
    })
    pred_df.to_csv(f'models/{scenario_name}/gnn_predictions_fixed.csv', index=False)
    
    scenario_results = {
        'Scenario': [scenario_name, scenario_name],
        'Model': ['Persistence', 'GNN'],
        'Horizon_Hours': [horizon, horizon],
        'Central Frequency 15cm': [pers_within15, gnn_within15],
        'MSE': [pers_mse, gnn_mse],
        'RMSE': [pers_rmse, gnn_rmse],
        'R2': [pers_r2, gnn_r2],
        'MAE': [pers_mae, gnn_mae],
        'MedAE': [pers_medae, gnn_medae],
        'MaxErr': [pers_maxerr, gnn_maxerr],
        'residual Stdev': [pers_residual_std, gnn_residual_std]
    }
    
    scenario_df = pd.DataFrame(scenario_results)
    scenario_df.to_csv(f'models/{scenario_name}/gnn_results_fixed.csv', index=False)
    
    all_results.append(scenario_df)
    
    print(f"\n  ✅ Saved: models/{scenario_name}/gnn_fixed.pth")

print("\n" + "="*80)
print("GNN RESULTS - ALL SCENARIOS (FIXED)")
print("="*80)

combined_df = pd.concat(all_results, ignore_index=True)
print("\n" + combined_df.to_string(index=False))

combined_df.to_csv('models/gnn_all_scenarios_results_fixed.csv', index=False)
print(f"\n✅ Saved: models/gnn_all_scenarios_results_fixed.csv")

