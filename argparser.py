import torch
import argparse

from .config import config_dict

def create_arg_parser():
    parser = argparse.ArgumentParser(description="Configuration for the model")
    
    parser.add_argument('--exp_num', type=str, default="custom", 
                        help='Experiment number from preset configs or "custom" to use custom arguments')
    parser.add_argument('--output_dir', type=str, default="custom", 
                        help='Directory to save model checkpoint')
    parser.add_argument('--exp_name', type=str, default="custom", 
                        help='Name of experiment')
    parser.add_argument('--model_name', type=str, default="mamba_enc_dec", choices=['mamba_seq2seq'],
                        help='Name of model to use')
    parser.add_argument('--seed', type=str, default=42,help='seed')
    
    # Data configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--src_max_len', type=int, default=300, help='Maximum length of source sequence')
    data_group.add_argument('--tgt_max_len', type=int, default=350, help='Maximum length of target sequence')
    data_group.add_argument('--truncate', action='store_true', default=False, help='Whether to truncate sequences')
    data_group.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Mamba configuration
    mamba_group = parser.add_argument_group('Mamba Configuration')
    mamba_group.add_argument('--enc_n_layer', type=int, default=5, help='Number of encoder layers')
    mamba_group.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    mamba_group.add_argument('--n_layer', type=int, default=5, help='Number of layers')
    mamba_group.add_argument('--rms_norm', action='store_true', default=True, help='Use RMS normalization')
    mamba_group.add_argument('--fused_add_norm', action='store_true', default=True, help='Use fused add norm')
    mamba_group.add_argument('--use_fast_path', action='store_true', default=False, help='Use fast path')
    mamba_group.add_argument('--mamba_learning_rate', type=float, default=5e-4, help='Learning rate for Mamba')
    mamba_group.add_argument('--mamba_warmup_steps', type=int, default=4000, help='Warmup steps for Mamba')
    mamba_group.add_argument('--mamba_weight_decay', type=float, default=0.001, help='Weight decay for Mamba')
    mamba_group.add_argument('--devices', type=str, default='cuda:0', help='Device for Mamba')
    
    # Trainer configuration
    trainer_group = parser.add_argument_group('Trainer Configuration')
    trainer_group.add_argument('--batch_size', type=int, default=16, help='Batch size')
    trainer_group.add_argument('--learning_rate', type=float, default=8e-4, help='Learning rate')
    trainer_group.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    trainer_group.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    trainer_group.add_argument('--grad_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    trainer_group.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm')
    trainer_group.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    trainer_group.add_argument('--device_id', type=int, default=1, help='Device ID')
    trainer_group.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs')
    trainer_group.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    
    #Test config
    testing_group = parser.add_argument_group('Tester Configuration')
    testing_group.add_argument('--test_limit', type=int, default=42, help='number of batches to infer on')
    return parser

def get_config():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Check if exp_num is "custom" or a preset number
    if args.exp_num.lower() == "custom":
        print("Using custom configuration from command-line arguments.")
        # Convert args to config dictionary
        config = {
            "model_name": args.model_name,
            "exp_num": args.exp_num,
            "exp_name": args.exp_name,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "data_config": {
                "src_max_len": args.src_max_len,
                "tgt_max_len": args.tgt_max_len,
                "truncate": args.truncate,
                "seed": args.seed
            },
            "mamba_config": {
                "enc_n_layer": args.enc_n_layer,
                "d_model": args.d_model,
                "n_layer": args.n_layer,
                "rms_norm": args.rms_norm,
                "fused_add_norm": args.fused_add_norm,
                "use_fast_path": args.use_fast_path,
                "learning_rate": args.mamba_learning_rate,
                "warmup_steps": args.mamba_warmup_steps,
                "weight_decay": args.mamba_weight_decay,
                "devices": args.devices
            },
            "trainer_config": {
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_epochs": args.num_epochs,
                "grad_accumulation_steps": args.grad_accumulation_steps,
                "max_grad_norm": args.max_grad_norm,
                "warmup_steps": args.warmup_steps,
                "device": torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu'),
                "device_id": args.device_id,
                "max_epochs": args.max_epochs,
                "patience": args.patience
            },
            "testing_config":{
                "test_limit": args.test_limit
            }
        }
    else:
        # Try to retrieve preset config
        if args.exp_num in config_dict:
            print(f"Using preset configuration for experiment {args.exp_num}.")
            config = config_dict[args.exp_num]
        else:
            available_presets = ", ".join(config_dict.keys())
            raise ValueError(f"Experiment number {args.exp_num} not found in preset configurations. "
                             f"Available presets are: {available_presets} or use 'custom'.")
    
    return config

