import os
import json
import logging

import pandas as pd

from .src.tokenizer import Tokenizer
from .src.Trainer import train_model
from .src.Evaluator import calculate_accuracy
from .src.models.model_factory import get_model
from .src.constants import special_symbols, BOS_IDX, EOS_IDX, PAD_IDX, SEP_IDX, UNK_IDX
from .src.data import get_dataloaders
from .argparser import parse_args

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Save arguments
    args_path = os.path.join(args.output_dir, args.exp_name, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Saved arguments to {args_path}")
    
    # Load model configuration
    with open(args.model_config, 'r') as f:
        model_config = json.load(f)
    

    # Call your existing train_model function with the parsed arguments
    logger.info(f"Starting training for {args.model_type} model")
    
    train_df = pd.read_csv('./data/train.csv')
    val_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/train.csv')

    data_config = None
    tokenizer = Tokenizer(train_df, args.index_token_pool_size, args.momentum_token_pool_size, special_symbols, UNK_IDX, False)
    dataloaders = get_dataloaders([train_df, val_df, test_df], \
                                  data_config, tokenizer, args.seed)
    
    model = train_model(model, dataloaders['train'], dataloaders['val'], max_epochs=10, 
               patience= 5, checkpoint_dir='checkpoints', device_id = 0)

    test_limit = args.test_limit if args.model_name == 'mamba_enc_dec' else None

    metrics = calculate_accuracy(model, dataloaders['test'], args.device, test_limit)
    # Save the trained model
    logger.info(f'metrics are {metrics}')
    
    model_path = os.path.join(args.output_dir, args.exp_name, 'final_model.pt')
    model.save(model_path)  # Assuming your model has a save method
    
    
    logger.info(f"Saved trained model to {model_path}")
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main()