import torch
import pickle

print("="*80)
print("DEBUGGING MODEL FILES")
print("="*80)

scenarios = ['Scenario1_1h', 'Scenario2_6h', 'Scenario3_12h']

for scenario in scenarios:
    print(f"\n{'='*80}")
    print(f"{scenario}")
    print('='*80)
    
    # Check LSTM
    print("\n[LSTM]")
    try:
        lstm_data = torch.load(f'models/{scenario}/lstm.pth', weights_only=False)
        
        if isinstance(lstm_data, dict):
            print(f"Type: Dictionary with {len(lstm_data)} keys")
            print(f"Keys: {list(lstm_data.keys())}")
            
            # Check for common wrapping patterns
            if 'model_state_dict' in lstm_data:
                actual_state = lstm_data['model_state_dict']
                print("✅ Wrapped in 'model_state_dict'")
            elif 'state_dict' in lstm_data:
                actual_state = lstm_data['state_dict']
                print("✅ Wrapped in 'state_dict'")
            else:
                actual_state = lstm_data
            
            # Print first few keys to see structure
            state_keys = list(actual_state.keys())
            print(f"\nFirst 10 state_dict keys:")
            for key in state_keys[:10]:
                print(f"  {key}: {actual_state[key].shape}")
            
            # Check LSTM architecture from weights
            if 'lstm.weight_ih_l0' in actual_state:
                input_size = actual_state['lstm.weight_ih_l0'].shape[1]
                hidden_size = actual_state['lstm.weight_ih_l0'].shape[0] // 4  # LSTM has 4 gates
                print(f"\n✅ Detected LSTM architecture:")
                print(f"   Input size: {input_size}")
                print(f"   Hidden size: {hidden_size}")
                
                # Count layers
                lstm_layers = [k for k in state_keys if k.startswith('lstm.weight_ih_l')]
                print(f"   Num layers: {len(lstm_layers)}")
            
            # Check final layer
            if 'fc.weight' in actual_state:
                fc_input = actual_state['fc.weight'].shape[1]
                fc_output = actual_state['fc.weight'].shape[0]
                print(f"   FC layer: {fc_input} → {fc_output}")
        
        else:
            print(f"Type: {type(lstm_data)}")
            print("✅ Full model object (not state_dict)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Check GNN
    print("\n[GNN]")
    try:
        gnn_data = torch.load(f'models/{scenario}/gnn_fixed.pth', weights_only=False)
        
        if isinstance(gnn_data, dict):
            print(f"Type: Dictionary with {len(gnn_data)} keys")
            print(f"Keys: {list(gnn_data.keys())}")
            
            # Check for common wrapping patterns
            if 'model_state_dict' in gnn_data:
                actual_state = gnn_data['model_state_dict']
                print("✅ Wrapped in 'model_state_dict'")
            elif 'state_dict' in gnn_data:
                actual_state = gnn_data['state_dict']
                print("✅ Wrapped in 'state_dict'")
            else:
                actual_state = gnn_data
            
            # Print keys
            state_keys = list(actual_state.keys())
            print(f"\nFirst 10 state_dict keys:")
            for key in state_keys[:10]:
                print(f"  {key}: {actual_state[key].shape}")
            
            # Check GNN architecture from weights
            if 'fc1.weight' in actual_state:
                input_size = actual_state['fc1.weight'].shape[1]
                hidden_size = actual_state['fc1.weight'].shape[0]
                print(f"\n✅ Detected GNN architecture:")
                print(f"   Input size: {input_size}")
                print(f"   Hidden size: {hidden_size}")
            
            if 'fc3.weight' in actual_state:
                output_size = actual_state['fc3.weight'].shape[0]
                print(f"   Output size: {output_size}")
        
        else:
            print(f"Type: {type(gnn_data)}")
            print("✅ Full model object (not state_dict)")
            
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)