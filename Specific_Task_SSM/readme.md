# Specific Task 3.2

## State Space Models

I have used Encoder-Decoder style Attention-SSM hybrid model, which has architecture similar to seq2seq transformer, replacing self-attention with Mamba kernels and keeping the cross attention intact.

## Implementation details

I have used official [mamba library](https://github.com/state-spaces/mamba/tree/main) and found an popular implementation on attention hybrids [github repo](https://github.com/deep-spin/ssm-mt).
I have used 1 Encoder and 1 Decoder block and achieved token_accuracy 94.68% and
sequence_accuracy 91.46%.

Model weights can be found at [link](https://drive.google.com/file/d/1fEdJcy9kPsqdHOKzVSz7nda-ZdkKR8yv/view?usp=sharing)

## Architecture

In proposal I have briefly discussed about two architectures that I'll work on. One of them being used here.
<p align='center'>
<img src='https://github.com/user-attachments/assets/18282c70-a9c4-4c50-946a-85c63e32d6f3' width="700">
</p>
