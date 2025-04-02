# Rationale on Common Task 1

Tokenization has huge impact on performance of the model. Amplitude and Squared Amplitude expressions consist of various terms like momentum, spinor indices and other mathematical symbols, which require careful tokenization to ensure meaningful input representations for the model.

1. Tokenizing Special Functions Properly

- Amplitude expressions often involve special functions like Γ (gamma function), δ (Dirac delta), and ε (Levi-Civita symbol). Tokenization should preserve these as atomic units rather than breaking them into individual characters.

- For example, δ(p - q) should be tokenized as a single function call rather than splitting δ, (, p, -, q, ) separately.

2. Normalizing Indices for Consistency

- Indices that are not Lorentz or spinor-related, such as summation indices or generic tensor indices, should be normalized to a consistent format.

- For instance, converting all summation indices (e.g., i, j, k) into a standardized form prevents the model from overfitting to specific index choices.

- Example: Instead of treating A_i in one sequence and A_j in another sequence as distinct structures, normalizing to a generic index representation (e.g., A_x) can improve generalization.

Write about Tokenization