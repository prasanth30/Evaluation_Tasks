import os
import argparse
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a model for research evaluation')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True, 
                        help='Directory containing the dataset')
    
    
    args = parser.parse_args()
    
    # Create experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f"{args.model_type}_{timestamp}"
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)
    
    return args