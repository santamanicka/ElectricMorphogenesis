#!/usr/bin/env python3
"""
Extract learned parameters from bestLearnedFacialParams_44.dat and save as JSON
for use in the JavaScript visualization.
"""

import torch
import json
import sys

def apply_sigmoid_constraint(raw_param, min_val, max_val):
    """Apply sigmoid transformation to get constrained value"""
    return min_val + (max_val - min_val) * torch.sigmoid(raw_param)

def extract_parameters(file_path):
    """Extract and transform learned parameters"""
    # Load the data
    data = torch.load(file_path, weights_only=False)

    params = {}
    param_bounds = data.get('parameter_bounds', {})

    # Extract each parameter
    for param_name, raw_value in data['parameters'].items():
        min_key = f'{param_name}_min'
        max_key = f'{param_name}_max'

        if min_key in param_bounds and max_key in param_bounds:
            # Apply sigmoid constraint
            constrained = apply_sigmoid_constraint(
                raw_value,
                param_bounds[min_key],
                param_bounds[max_key]
            )
            params[param_name] = float(constrained.item())
        else:
            # Fallback to raw value
            params[param_name] = float(raw_value.item())

    # Add metadata
    result = {
        'parameters': params,
        'loss': float(data['loss']),
        'grid_size': int(data['grid_size']),
        'learned_parameter_names': data['learned_parameter_names']
    }

    return result

if __name__ == '__main__':
    # Default to file 44, or use command line argument
    file_number = int(sys.argv[1]) if len(sys.argv) > 1 else 44

    input_file = f'data/bestLearnedFacialParams_{file_number}.dat'
    output_file = f'data/learned_params_{file_number}.json'

    print(f"Extracting parameters from {input_file}...")

    try:
        result = extract_parameters(input_file)

        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"✓ Parameters saved to {output_file}")
        print(f"\nExtracted {len(result['parameters'])} parameters:")
        for name, value in sorted(result['parameters'].items()):
            print(f"  {name:30s}: {value:.4f}")

        print(f"\nLoss: {result['loss']:.6f}")
        print(f"Grid size: {result['grid_size']}")

    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)