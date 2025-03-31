from mamba_hybrid import MambaEncDec


def get_model(model_name, args):

    if model_name == 'seq2seq_transformer':
        return 
    
    elif model_name == 'mamba_enc_dec':
        return MambaEncDec(**args, config=args)
    
    else:
        raise "Model not found"