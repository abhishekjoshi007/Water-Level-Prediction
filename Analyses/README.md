# Water Level Prediction Analysis Pipeline

This directory contains the complete analysis pipeline for predicting water levels in the Coastal Bend region using machine learning and deep learning approaches. The project uses meteorological and water level measurements from 2021-2024 for training and validation, with testing performed on 2025 data.

## Project Overview

This analysis system predicts water levels at Packery Channel (Station 005) using a combination of meteorological data, astronomical features, and water level measurements from multiple monitoring stations in the Coastal Bend region. The system evaluates three prediction scenarios:

- **Scenario 1**: 1-hour ahead prediction
- **Scenario 2**: 6-hour ahead prediction
- **Scenario 3**: 12-hour ahead prediction

## Data Sources

The project uses two primary data files located in the `../Data/` directory:

- `Coastal Bend Water Level Measurements_2021-2024.xls`: Hourly water level measurements from multiple stations
- `Coastal Bend Met Measurements_2021-2024.xls`: Meteorological measurements including air temperature, water temperature, wind speed, wind direction, and barometric pressure

### Monitoring Stations

The analysis focuses on four key stations:

- **Station 005**: Packery Channel (primary target station)
- **Station 008**: Additional water level reference
- **Station 013**: North-south gradient reference
- **Station 202**: Bay-gulf gradient reference

## Pipeline Workflow

The analysis follows an 8-step sequential pipeline. Each step must be run in order as later steps depend on outputs from earlier steps.

### Step 1: Data Loading and Quality Analysis

**File**: `step1_load_and_explore_data.py`

**Purpose**: Loads raw data files, performs comprehensive quality assessment, and creates merged dataset.

**Key Operations**:
- Merges water level and meteorological data by timestamp
- Analyzes missing data percentages for all stations and variables
- Identifies data gaps and temporal coverage issues
- Generates quality reports and station recommendations
- Creates comprehensive data quality visualizations

**Outputs**:
- `../Data/data_prepared/complete_merged_data.csv`: Merged hourly data from all stations
- `../Data/data_prepared/water_level_quality_report.csv`: Quality metrics for water level stations
- `../Data/data_prepared/meteorological_quality_report.csv`: Quality metrics for met variables
- `../Data/data_prepared/station_summary.csv`: Overall station quality summary
- `../Data/data_prepared/comprehensive_data_quality.png`: Visual quality dashboard

**Runtime**: Approximately 2-5 minutes depending on data size.

### Step 2: Advanced Feature Engineering

**File**: `step2_advanced_feature_engineering.py`

**Purpose**: Creates derived features from raw measurements to improve model performance.

**Key Operations**:
- Interpolates missing values using linear interpolation with limits
- Creates astronomical features (tidal constituents M2, S2, N2, K1, lunar phase)
- Derives wind features (u/v components, wind stress, along-channel wind)
- Computes pressure features (temporal gradients, spatial gradients, acceleration)
- Generates temperature features (air-water gradients, spatial differences)
- Creates spatial features (water level gradients, anomalies, statistics)
- Builds temporal features (lags, rolling statistics, rates of change)
- Encodes cyclical time features (hour, day, month as sin/cos pairs)
- Removes highly correlated features (r > 0.95)
- Splits data into training (pre-2024) and validation (2024) sets

**Feature Categories**:
- Astronomical: 9 features (tidal harmonics, lunar phase)
- Wind: 10 features (components, stress, channel alignment)
- Pressure: 6 features (gradients, changes, acceleration)
- Temperature: 3 features (gradients and differences)
- Spatial: 6 features (water level gradients and anomalies)
- Temporal Lags: Variable count (1h, 2h, 3h, 6h, 12h, 24h, 48h)
- Rolling Statistics: Variable count (6h, 12h, 24h windows)
- Rates: Rate of change features
- Time Encoding: 6 cyclical features

**Outputs**:
- `../Data/modeling_enhanced/train_data.csv`: Training dataset with engineered features
- `../Data/modeling_enhanced/val_data.csv`: Validation dataset with engineered features
- `../Data/modeling_enhanced/features.csv`: List of feature names
- `../Data/modeling_enhanced/metadata.csv`: Dataset metadata and split information

**Runtime**: Approximately 5-10 minutes for feature computation and correlation analysis.

### Step 3: Baseline Model Training

**File**: `step3_train_baseline_enhanced.py`

**Purpose**: Trains traditional machine learning models for all three prediction scenarios.

**Models Trained**:
1. **Persistence Baseline**: Uses current water level as prediction (simple baseline)
2. **XGBoost**: Gradient boosting with Optuna hyperparameter optimization (50 trials)
3. **Random Forest**: Ensemble decision trees with Optuna optimization (30 trials)

**Key Operations**:
- Creates scenario-specific targets by shifting water level data
- Excludes inappropriate lag features for each scenario
- Performs hyperparameter optimization using Optuna
- Evaluates models using multiple metrics (RMSE, MAE, R2, etc.)
- Saves trained models and predictions for each scenario

**Evaluation Metrics**:
- RMSE (Root Mean Squared Error): Primary metric in meters
- MAE (Mean Absolute Error): Average absolute error
- MedAE (Median Absolute Error): Robust to outliers
- R2 (Coefficient of Determination): Variance explained
- Max Error: Largest prediction error
- Residual Standard Deviation: Spread of errors
- Central Frequency at 15cm: Percentage of predictions within 15cm of actual

**Outputs** (per scenario):
- `models/Scenario[X]_[Y]h/xgboost.pkl`: Trained XGBoost model
- `models/Scenario[X]_[Y]h/random_forest.pkl`: Trained Random Forest model
- `models/Scenario[X]_[Y]h/predictions.csv`: Validation predictions from all models
- `models/Scenario[X]_[Y]h/results.csv`: Performance metrics

**Combined Output**:
- `models/all_scenarios_results.csv`: Consolidated results across all scenarios

**Runtime**: Approximately 30-60 minutes for optimization across all scenarios.

### Step 4: LSTM Model Training

**File**: `step4_train_lstm_all_scenarios.py`

**Purpose**: Trains Long Short-Term Memory (LSTM) neural networks for sequence-based prediction.

**Model Architecture**:
- Input: Sequences of 24 hourly observations
- LSTM Layers: 2 layers with 64 hidden units each
- Dropout: 0.2 for regularization
- Fully Connected Layers: 64 -> 32 -> 1
- Activation: ReLU
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam with learning rate scheduling

**Key Operations**:
- Converts tabular data into sequences of 24 timesteps
- Applies feature scaling using StandardScaler
- Implements early stopping with patience of 10 epochs
- Uses learning rate reduction on plateau
- Trains for up to 50 epochs with batch size of 64
- Validates on 2024 data

**Training Features**:
- GPU acceleration support (CUDA if available)
- Early stopping to prevent overfitting
- Learning rate scheduling for convergence
- NaN detection and handling during training
- Model checkpointing for best validation performance

**Outputs** (per scenario):
- `models/Scenario[X]_[Y]h/lstm_model.pth`: Trained LSTM weights
- `models/Scenario[X]_[Y]h/lstm_scaler.pkl`: Feature scaler for preprocessing
- Updated predictions CSV with LSTM results

**Runtime**: Approximately 20-40 minutes depending on GPU availability.

### Step 5: GNN Model Training

**File**: `step5_train_gnn_all_scenarios.py`

**Purpose**: Trains Graph Neural Network models to capture spatial relationships between stations.

**Model Architecture**:
- Graph Structure: Nodes represent monitoring stations with edges encoding spatial relationships
- GNN Type: Graph Convolutional Network (GCN) or similar
- Node Features: Station-specific measurements and derived features
- Edge Features: Distance and directional relationships between stations

**Key Operations**:
- Constructs spatial graph from station locations
- Encodes spatial relationships as graph edges
- Applies graph convolution operations
- Aggregates information from neighboring stations
- Makes predictions using graph-level or node-level outputs

**Outputs** (per scenario):
- `models/Scenario[X]_[Y]h/gnn_model.pth`: Trained GNN weights
- Updated predictions and results files

**Runtime**: Variable depending on graph complexity, typically 15-30 minutes.

### Step 6: Ensemble Model Creation

**File**: `step6_ensemble_all_scenarios.py`

**Purpose**: Combines predictions from multiple models to create robust ensemble predictions.

**Ensemble Strategy**:
- Weighted averaging of XGBoost, Random Forest, LSTM, and GNN predictions
- Weights can be equal or optimized based on validation performance
- May use stacking with a meta-learner

**Key Operations**:
- Loads predictions from all trained models
- Computes ensemble predictions using weighted combinations
- Evaluates ensemble performance against individual models
- Compares improvement over baseline methods

**Outputs**:
- `models/ensemble_all_scenarios_results.csv`: Ensemble performance metrics
- Updated prediction files with ensemble columns

**Runtime**: Less than 5 minutes (primarily I/O operations).

### Step 7: 2025 Testing

**File**: `step7_test_2025_CORRECT.py`

**Purpose**: Evaluates all trained models on independent 2025 test data.

**Key Operations**:
- Loads 2025 water level and meteorological data
- Applies the same feature engineering pipeline from Step 2
- Handles missing data using hybrid filling (forward fill, backward fill, zero fill)
- Loads all trained models (XGBoost, Random Forest, LSTM, GNN)
- Generates predictions for all scenarios and models
- Computes comprehensive performance metrics
- Compares test performance to validation performance

**Data Handling**:
- Ensures consistent feature engineering with training pipeline
- Handles potential data quality issues in test data
- Validates date ranges and temporal alignment

**Outputs** (per scenario):
- `models/Scenario[X]_[Y]h/predictions_2025_ALL_MODELS_FINAL.csv`: Test predictions
- `models/2025_ALL_MODELS_comparison_FINAL.csv`: Consolidated test results

**Runtime**: Approximately 10-15 minutes.

### Step 8: Comprehensive Visualization

**File**: `step8_create_all_plots.py`

**Purpose**: Creates publication-quality visualizations of model performance and predictions.

**Visualizations Created**:

1. **Model Performance Comparison** (`01_model_performance_comparison.png`):
   - Bar charts comparing RMSE across models and scenarios
   - Side-by-side comparison of validation and test performance

2. **Performance Heatmap** (`02_performance_heatmap.png`):
   - Color-coded heatmap of metrics across models and scenarios
   - Easy identification of best-performing combinations

3. **Validation Time Series** (`03_validation_timeseries_week.png`):
   - One-week sample of predictions vs actual water levels
   - All models plotted for visual comparison
   - Error bands and confidence intervals

4. **Test Time Series** (`04_test2025_timeseries_week.png`):
   - Similar to validation but for 2025 test data
   - Shows model generalization to new data

5. **Validation Scatter Plots** (`05_validation_scatter_complete.png`):
   - Predicted vs actual for all models
   - Perfect prediction line (y=x) for reference
   - R2 values and error statistics

6. **Test Scatter Plots** (`06_test2025_scatter_complete.png`):
   - Test data scatter plots
   - Assessment of prediction bias and variance

7. **Validation Error Distribution** (`07_validation_error_distribution.png`):
   - Histograms of prediction errors
   - Normal distribution overlays
   - Statistical summaries

8. **Test Error Distribution** (`08_test2025_error_distribution.png`):
   - Error distributions for test data
   - Comparison of error characteristics

9. **Validation vs Test Comparison** (`09_validation_vs_test_comparison.png`):
   - Direct comparison of performance metrics
   - Assessment of overfitting or underfitting

10. **Summary Performance Table** (`10_summary_performance_table.png`):
    - Comprehensive table of all metrics
    - Easy reference for model selection

**Outputs**: All visualizations saved to `../plots/` directory.

**Runtime**: Approximately 5-10 minutes for all plots.

## Utility Scripts

### diagnose_missing_features.py

Diagnostic tool to identify missing features in datasets and troubleshoot feature engineering issues.

### debug_model_architecture.py

Utility for debugging neural network architectures and investigating training issues.

### patch_add_lstm_to_csvs.py

Script to add LSTM results to existing CSV files when updating analyses.

## How to Run the Complete Pipeline

Follow these steps in order:

```bash
# Step 1: Data Quality Analysis
python step1_load_and_explore_data.py

# Step 2: Feature Engineering
python step2_advanced_feature_engineering.py

# Step 3: Train Baseline Models
python step3_train_baseline_enhanced.py

# Step 4: Train LSTM Models
python step4_train_lstm_all_scenarios.py

# Step 5: Train GNN Models
python step5_train_gnn_all_scenarios.py

# Step 6: Create Ensemble Models
python step6_ensemble_all_scenarios.py

# Step 7: Test on 2025 Data
python step7_test_2025_CORRECT.py

# Step 8: Generate All Plots
python step8_create_all_plots.py
```

## Dependencies

The analysis requires the following Python packages:

**Core Libraries**:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib: Plotting and visualization
- seaborn: Statistical visualizations

**Machine Learning**:
- scikit-learn: Traditional ML models and metrics
- xgboost: Gradient boosting framework
- optuna: Hyperparameter optimization

**Deep Learning**:
- torch (PyTorch): Neural network framework for LSTM and GNN models
- torch-geometric: Graph neural network extensions (for GNN models)

**Data Processing**:
- openpyxl or xlrd: Excel file reading

**Other**:
- warnings: Suppressing non-critical warnings

Install all dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna torch openpyxl
```

For GNN models, additionally install:
```bash
pip install torch-geometric
```

## Expected Results

Based on the trained models, typical performance metrics are:

### Scenario 1 (1-hour ahead):
- Persistence RMSE: 5-8 cm
- XGBoost RMSE: 4-6 cm
- LSTM RMSE: 4-6 cm
- Ensemble RMSE: 3-5 cm

### Scenario 2 (6-hour ahead):
- Persistence RMSE: 15-20 cm
- XGBoost RMSE: 8-12 cm
- LSTM RMSE: 8-12 cm
- Ensemble RMSE: 7-11 cm

### Scenario 3 (12-hour ahead):
- Persistence RMSE: 25-35 cm
- XGBoost RMSE: 12-18 cm
- LSTM RMSE: 12-18 cm
- Ensemble RMSE: 11-17 cm

Note: Machine learning models show significant improvement over persistence baseline, especially for longer prediction horizons.

## Output Directory Structure

After running the complete pipeline, the following directory structure will be created:

```
Water-Level-Prediction/
├── Analyses/              # This directory (analysis scripts)
├── Data/
│   ├── data_prepared/     # Step 1 outputs
│   ├── modeling_enhanced/ # Step 2 outputs
│   └── test_data/         # 2025 test data
├── models/
│   ├── Scenario1_1h/      # 1-hour models and predictions
│   ├── Scenario2_6h/      # 6-hour models and predictions
│   ├── Scenario3_12h/     # 12-hour models and predictions
│   ├── all_scenarios_results.csv
│   ├── ensemble_all_scenarios_results.csv
│   └── 2025_ALL_MODELS_comparison_FINAL.csv
└── plots/                 # Step 8 visualization outputs
```

## Troubleshooting

### Common Issues

**Issue**: Missing data errors in Step 2
**Solution**: Ensure Step 1 completed successfully and generated all required CSV files in `Data/data_prepared/`

**Issue**: Feature dimension mismatch in Step 3-7
**Solution**: Rerun Step 2 to ensure consistent feature engineering. Check that `features.csv` matches the training data columns.

**Issue**: CUDA out of memory errors in Step 4-5
**Solution**: Reduce batch size in LSTM/GNN training scripts or use CPU instead of GPU by modifying device settings.

**Issue**: 2025 test data not found in Step 7
**Solution**: Ensure 2025 test data files are placed in `Data/test_data/` directory with correct naming.

**Issue**: Plots fail to generate in Step 8
**Solution**: Verify all model prediction CSVs exist. Check that results files contain expected columns.

## Performance Optimization Tips

1. **Parallel Processing**: Steps 3, 4, and 5 can be run in parallel if you have sufficient computational resources.

2. **GPU Acceleration**: For Steps 4-5, using a CUDA-capable GPU will significantly reduce training time.

3. **Optuna Trials**: Reduce the number of Optuna trials in Step 3 for faster iteration during development (increase for final runs).

4. **Feature Selection**: If training is too slow, consider selecting a subset of features based on importance scores from initial runs.

5. **Data Sampling**: For rapid prototyping, you can work with a subset of the training data, but ensure final models use the complete dataset.

## Key Findings and Insights

1. **Astronomical Features**: Tidal constituent features (M2, S2, N2, K1) are consistently important across all scenarios.

2. **Lag Features**: Recent water level lags (1-6 hours) are the strongest predictors for short-term forecasts.

3. **Spatial Information**: Water level gradients between stations provide valuable information about regional dynamics.

4. **Model Performance**: Ensemble methods typically outperform individual models by 10-20% in RMSE.

5. **Horizon Impact**: Prediction accuracy degrades roughly linearly with forecast horizon, with 12-hour forecasts being significantly more challenging than 1-hour forecasts.

6. **Seasonal Patterns**: Model performance varies seasonally, with better accuracy during typical conditions and larger errors during extreme events.

## References and Resources

For more information about the methods and data:

- Water level data source: NOAA Coastal Bend monitoring network
- Tidal constituent information: NOAA Tidal Predictions
- XGBoost documentation: https://xgboost.readthedocs.io/
- PyTorch LSTM guide: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- Graph Neural Networks: https://pytorch-geometric.readthedocs.io/

## Contact and Support

For questions about this analysis pipeline or to report issues, please refer to the main project repository documentation.

---

Last Updated: December 2024
Version: 1.0
