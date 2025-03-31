import random
from tqdm.auto import tqdm
import pandas as pd

def normalize_indices(tokenizer, expressions, index_token_pool_size=50, momentum_token_pool_size=50):
    # Function to replace indices with a new set of tokens for each expression
    def replace_indices(token_list, index_map):
        new_index = (f"_{i}" for i in range(index_token_pool_size))  # Local generator for new indices
        new_tokens = []
        for token in token_list:
            if "INDEX_" in token:
                if token not in index_map:
                    try:
                        index_map[token] = token.rsplit('_',1)[0] + next(new_index)
                    except StopIteration:
                        # Handle the case where no more indices are available
                        raise ValueError("Ran out of unique indices, increase token_pool_size")
                new_tokens.append(index_map[token])
            else:
                new_tokens.append(token)
        return new_tokens

    def replace_momenta(token_list, index_map):
        new_index = (f"MOMENTUM_{i}" for i in range(momentum_token_pool_size))  # Local generator for new indices
        new_tokens = []
        for token in token_list:
            if "MOMENTUM_" in token:
                if token not in index_map:
                    try:
                        index_map[token] = next(new_index)
                    except StopIteration:
                        # Handle the case where no more indices are available
                        raise ValueError("Ran out of unique indices, increase momentum_token_pool_size")
                new_tokens.append(index_map[token])
            else:
                new_tokens.append(token)
        return new_tokens

    normalized_expressions = []
    # Replace indices in each expression randomly
    for expr in tqdm(expressions,desc="Normalizing.."):
        toks = tokenizer.src_tokenize(expr,42)
        normalized_expressions.append(replace_momenta(replace_indices(toks, {}), {}))

    return normalized_expressions


def aug_data(df, tokenizer):
    # Extract columns
    amps = df['Amplitude']
    sqamps = df['Squared Amplitude']

    # Data augmentation
    n_samples = 1 #args.n_samples
    aug_amps = []

    for amp in tqdm(amps, desc='processing'):
        random_seed = [random.randint(1, 1000) for _ in range(n_samples)]
        for seed in random_seed:
            aug_amps.append(tokenizer.src_replace(amp, seed))
    aug_sqamps = [sqamp for sqamp in sqamps for _ in range(n_samples)]

    if True:
        normal_amps = normalize_indices(tokenizer, aug_amps, 500, 500)
        aug_amps = []
        for amp in normal_amps:
            aug_amps.append("".join(amp))

    # Create augmented DataFrame
    df_aug = pd.DataFrame({"Amplitude": aug_amps, "Squared Amplitude": aug_sqamps})

    return df_aug