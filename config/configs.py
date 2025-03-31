import torch

config_dict = {
    
}

config = {
    "mamba_enc_dec":{
        "model_args":{

        },

    },
    "seq2seq_transformer":{
        "model_args"
    },
}

default_config = {
    "data_config":{
        "src_max_len": 300,
        "tgt_max_len": 350,
        "truncate": False,
        "seed": 42
    },
    "mamba_config":{
        "enc_n_layer": 5,
        "d_model": 512,
        "n_layer": 5,  
        "rms_norm": True,
        "fused_add_norm": True,
        "use_fast_path": False,
        "learning_rate": 5e-4,
        "warmup_steps": 4000,
        "weight_decay": 0.001,
        "devices": 'cuda:0'
    },
    "trainer_config":{
        "batch_size" : 16,
        "learning_rate" : 8e-4,
        "weight_decay" : 0.01,
        "num_epochs" : 50,
        "grad_accumulation_steps" : 8,
        "max_grad_norm" : 1.0,
        "warmup_steps" : 1000,
        "device" : torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu'),
        "device_id": 1,
    },
    "training_config":{
        "max_epochs": 50,
        "patience": 5
    }
}




"""
Config required
tokenizer:
index_token_pool_size, momentum_token_pool_size

dataset config:

src_max_len = 300
tgt_max_len = 350
truncate = False
seed = 42

"default": {
    "enc_n_layer": 5,
    # mamba config
    "d_model": 512,
    "n_layer": 5,
    "rms_norm": True,
    "fused_add_norm": True,
    "use_fast_path": False,
    "learning_rate": 5e-4,
    "warmup_steps": 4000,
    "weight_decay": 0.001,
    "devices": 'cuda:0'
}
training:

batch_size = 16
learning_rate = 8e-4
weight_decay = 0.01
num_epochs = 50
grad_accumulation_steps = 8
max_grad_norm = 1.0
warmup_steps = 1000
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


max_epochs=50,
patience=51,

testing:
max_limit


"""