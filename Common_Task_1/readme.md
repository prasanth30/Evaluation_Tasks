# Common Task 1
Description:

Download the dataset (split across 10 files) and preprocess and tokenize the target data and document your rationale for choice of tokenization. Data file is formatted with rows like 
“event type : Feynman diagram : amplitude : squared amplitude”
Here the amplitudes are the input sequences and squared amplitudes are the target sequences. Note that indices like _123456 grow over the course of the dataset and should be normalized for each amplitude and squared amplitude. Use an 80-10-10 split of train-val-test across all files.

## Rationale on Common Task 1

Tokenization has huge impact on performance of the model. Amplitude and Squared Amplitude expressions consist of various terms like momentum, spinor indices and other mathematical symbols, which require careful tokenization to ensure meaningful input representations for the model.

1. Tokenizing Special Functions Properly

- Amplitude expressions often involve special functions like Γ (gamma function), δ (Dirac delta), and ε (Levi-Civita symbol). Tokenization should preserve these as atomic units rather than breaking them into individual characters.

2. Normalizing Indices for Consistency

- Indices that are not Lorentz or spinor-related, such as summation indices or generic tensor indices, should be normalized to a consistent format.

## Implementation details

- I have used the text files to extract the expressions and create dataframe. Which was further split it into train-val-test split. I have normalized the expressions and built tokenizer.

- To simplify implementation, I have used vocab class from torchtext (which got deprecated, so I have rebuilt it from source)

Refer to the notebook for more details