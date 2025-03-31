# SYMBA_SSM_TASKS

## Overview
This repo is my submission of evaluation tasks for **State-space models for squared amplitude calculation in high-energy physics** SYMBA, ML4Sci.

## Project Structure

```
SYMBA_SSM_TASKS/
│-- config/                  # Configuration files
│   │-- __init__.py          
│   │-- configs.py           # Defines various model and training configurations
│   │-- get_config.py        # Utility for fetching configurations
│
│-- data/                    # Directory for datasets (empty by default)
│
│-- notebooks/               # Jupyter notebooks for experiments (empty by default)
│
│-- src/                     # Source code for models and utilities
│   │-- models/              # Model implementations
│   │   │-- mamba_hybrid/    # Hybrid Mamba-based models
│   │   │   │-- utils/helpers/
│   │   │   │   │-- cross_attention.py
│   │   │   │   │-- ffn.py
│   │   │   │   │-- flash_cross_attention.py
│   │   │   │   │-- __init__.py
│   │   │   │-- mamba_enc_dec.py  # Mamba-based encoder-decoder model
│   │   │-- mamba.py         # Core Mamba model implementation
│   │   │-- transformer_seq2seq/ # Transformer-based seq2seq models
│   │   │-- model_factory.py  # Factory function for model selection
│   │
│   │-- utils/               # General-purpose utilities
│   │   │-- Vocab/           # Vocabulary utilities
│   │   │   │-- __init__.py
│   │   │-- constants.py     # Constant definitions
│   │   │-- data.py          # Data handling functions
│   │   │-- Evaluator.py     # Evaluation metrics and scoring
│   │   │-- preprocess_data.py # Data preprocessing scripts
│   │   │-- preprocess.py    # General preprocessing functions
│   │   │-- tokenizer.py     # Tokenization utilities
│   │   │-- Trainer.py       # Training pipeline
```

## Setup

### Prerequisites
This implementation consists of mamba_ssm library which depends on hardware type used so make sure you have a GPU with CUDA 11.6 and above. If you face errors still checkout, [This Issue](https://github.com/state-spaces/mamba/issues/186). For Instance on kaggle for T4x2, this code can be run effortlessly with proper installations, but P100 doesnot work and requires some troubleshooting.

```sh
conda create -n symba_ssm python=3.10 -y
conda activate symba_ssm
```

### Install Dependencies

You can install the required dependencies using:

Install mamba_ssm from source (takes around 3 minutes)
```sh
git clone https://github.com/state-spaces/mamba.git
pip install -q /kaggle/working/mamba
```
Other Requirements (takes around 1 minute)
```sh
pip install -q causal-conv1d>=1.4.0
pip install triton
pip -q install lightning torchscale evaluate huggingface_hub unbabel-comet flash-attn
pip install -q x-transformers
```

## Usage

### Running Training
To start model training, run specific config saved in config_dict of `config/configs.py`
```sh
python train.py --exp_num 5
```
To give config in command line use
```sh
python train.py --exp_num custom
```


