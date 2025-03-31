import torch
from torch.utils.data import Dataset, DataLoader

from .constants import EOS_IDX, BOS_IDX, PAD_IDX
def causal_mask(size):
    """Create a causal mask for a sequence of given size."""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).int()
    return mask == 0
    
class Data(Dataset):
    """
    Custom PyTorch dataset for handling data.

    Args:
        df (DataFrame): DataFrame containing data.
    """

    def __init__(self, df, tokenizer, config, src_vocab, tgt_vocab):
        super(Data, self).__init__()
        self.tgt_vals = df['Squared Amplitude']
        self.src_vals = df['Amplitude']
        self.tgt_tokenize = tokenizer.tgt_tokenize
        self.src_tokenize = tokenizer.src_tokenize
        self.bos_token = torch.tensor([BOS_IDX], dtype=torch.int64)
        self.eos_token = torch.tensor([EOS_IDX], dtype=torch.int64)
        self.pad_token = torch.tensor([PAD_IDX], dtype=torch.int64)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.src_vals)

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing source and target tensors.
        """
        # print(f'index: {idx}')
        src_tokenized = self.src_tokenize(self.src_vals[idx],self.config.seed)
        tgt_tokenized = self.tgt_tokenize(self.tgt_vals[idx])
        src_ids = self.src_vocab.forward(src_tokenized)
        tgt_ids = self.tgt_vocab.forward(tgt_tokenized)

        enc_num_padding_tokens = self.config.src_max_len - len(src_ids) - 2
        dec_num_padding_tokens = self.config.tgt_max_len - len(tgt_ids) - 1
        # print(f'src_ids: {len(src_ids)} tgt_ids:  {len(tgt_ids)} enc_num: {enc_num_padding_tokens} dec_num: {dec_num_padding_tokens} \n' )
        if self.config.truncate:
            if enc_num_padding_tokens < 0:
                src_ids = src_ids[:self.config.src_max_len-2]
            if dec_num_padding_tokens < 0:
                tgt_ids = tgt_ids[:self.config.tgt_max_len-1]
        else:
            if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
                raise ValueError("Sentence is too long")
        src_tensor = torch.cat(
            [
                self.bos_token,
                torch.tensor(src_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] *
                             enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        tgt_tensor = torch.cat(
            [
                self.bos_token,
                torch.tensor(tgt_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] *
                             dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(tgt_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        src_mask = (src_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int() # (1, 1, seq_len)
        tgt_mask = (tgt_tensor != self.pad_token).unsqueeze(0).int() & causal_mask(tgt_tensor.size(0)) # (1, seq_len) & (1, seq_len, seq_len),

        return src_tensor, tgt_tensor, label, src_mask, tgt_mask#, len(src_ids), len(tgt_ids)

    @staticmethod
    def get_data(df_train, df_valid, df_test, config, tokenizer, src_vocab, tgt_vocab):
        """
        Create datasets (train, test, and valid)

        Returns:
            dict: Dictionary containing train, test, and valid datasets.
        """
        train = Data(df_train, tokenizer, config,src_vocab,tgt_vocab)
        test = Data(df_test, tokenizer, config,src_vocab,tgt_vocab)
        valid = Data(df_valid, tokenizer, config,src_vocab,tgt_vocab)

        return {'train': train, 'test': test, 'valid': valid}
    


def get_dataloaders(df_list, data_config, tokenizer, seed):
    """
    args
    df_list: [train_df, val_df, test_df]
    data_config: config class
    tokenizer: tokenizer to use
    seed: seed for building vocab
    """
    train_df, val_df, test_df = df_list
    src_vocab2 = tokenizer.build_src_vocab(seed)
    tgt_vocab2 = tokenizer.build_tgt_vocab(seed)
    dataset = Data.get_data(train_df, val_df, test_df, data_config, tokenizer, src_vocab2, tgt_vocab2)
    dataloaders = {
        split: DataLoader(
            dataset[split],
            batch_size= data_config["batch_size"] if split != 'test' else 4,
            shuffle= (split == 'train'),
            pin_memory= True,
            num_workers= 4
        ) for split in ['train', 'valid', 'test']
    }

    return dataloaders