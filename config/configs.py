config = {
    "mamba_enc_dec":{
        "model_args":{

        },

    },
    "seq2seq_transformer":{
        "model_args"
    },
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







"""