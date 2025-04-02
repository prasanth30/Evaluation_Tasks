# Specific Task 3.2

## State Space Models

I have used Encoder-Decoder style Attention-SSM hybrid model, which has architecture similar to seq2seq transformer, replacing self-attention with Mamba kernels and keeping the cross attention intact.

## Implementation details

I have used official [mamba library](https://github.com/state-spaces/mamba/tree/main) and found an popular implementation on attention hybrids [github repo](https://github.com/deep-spin/ssm-mt).
I have used 1 Encoder and 1 Decoder block and achieved token_accuracy 94.68% and
sequence_accuracy 91.46%.