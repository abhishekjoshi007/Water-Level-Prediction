import pandas as pd
import numpy as np

print("="*80)
print("PATCHING CSV FILES TO ADD LSTM COLUMN")
print("="*80)

scenarios = ['Scenario1_1h', 'Scenario2_6h', 'Scenario3_12h']

for scenario in scenarios:
    print(f"\n[{scenario}]")
    
    try:
        # Load existing predictions
        df = pd.read_csv(f'models/{scenario}/predictions_2025_ALL_MODELS_FINAL.csv')
        
        # Add LSTM column with NaN
        if 'lstm' not in df.columns:
            df['lstm'] = np.nan
            print(f"  ✅ Added LSTM column ({len(df)} rows)")
        else:
            print(f"  ℹ️  LSTM column already exists")
        
        # Reorder columns
        column_order = ['datetime', 'actual', 'persistence', 'xgboost', 'random_forest', 'lstm', 'ensemble']
        existing_cols = [c for c in column_order if c in df.columns]
        df = df[existing_cols]
        
        # Save
        df.to_csv(f'models/{scenario}/predictions_2025_ALL_MODELS_FINAL.csv', index=False)
        print(f"  💾 Saved with columns: {', '.join(df.columns)}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*80)
print("✅ DONE!")
print("="*80)
print("\nNext: Run step11_create_final_plots.py for complete visualizations")