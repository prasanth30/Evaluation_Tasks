# SYMBA_SSM_TASKS

## Overview

This repo is my submission of evaluation tasks for **State-space models for squared amplitude calculation in high-energy physics** SYMBA, ML4Sci.

## Folders

1. Folder [Common Task 1](./Common_Task_1/) contains notebook for tokenization and rationale of choice.
2. Folder [Common Task 2](./Common_Task_2/) contains notebook for Transformer Architecture.
3. Folder [Specific Task 3.2](./Specific_Task_SSM/) contains notebook with SSMs.

## Problem Statement

Squared Amplitudes play a major role in calculation cross-section or probabilty that a particular process takes place in the interaction of elementary particles. Using Amplitude expressions one can use a Seq2Seq model to get squared Amplitude expressions.

## Common Task 1

Dataset preprocessing
Dataset: [Link](https://alabama.box.com/s/xhgr2onrn503jyse2fs5vxtapg0oifcs) 

For Details: [Readme](./Common_Task_1/readme.md)

## Common Task 2

Training a generic next-token-prediction Transformer model to map the input data to the tokenized output sequences.

For details and model weights: [Readme](./Common_Task_2/readme.md)

## Specific Task - State Space Models

State-space model such as mamba or other model for squared amplitudes calculation

For details and model weights: [Readme](./Specific_Task_SSM/readme.md)
## Project Structure

```
SYMBA_SSM_TASKS/
├── 📂 Common_Task_1
│   ├── 📄 readme.md
│   ├── 📄 Tokenization&EDA.ipynb
├── 📂 Common_Task_2
│   ├── 📄 readme.md
│   ├── 📄 transformer.ipynb
├── 📂 config
│   ├── 🐍 __init__.py
│   ├── 🐍 configs.py
│   ├── 🐍 get_config.py
├── 📂 data
├── 📂 Specific_Task_SSM
│   ├── 📄 readme.md
│   ├── 📄 ssm_testing.ipynb
│   ├── 📄 ssm_training.ipynb
├── 📂 src
│   ├── 📂 models
│   │   ├── 📂 mamba_hybrid
│   │   │   ├── 📂 utils
│   │   │   │   ├── 📂 helpers
│   │   │   │   │   ├── 🐍 __init__.py
│   │   │   │   │   ├── 🐍 cross_attention.py
│   │   │   │   │   ├── 🐍 ffn.py
│   │   │   │   │   ├── 🐍 flash_cross_attention.py
│   │   │   ├── 🐍 mamba.py
│   │   │   ├── 🐍 mamba_enc_dec.py
│   │   ├── 📂 transformer_seq2seq
│   │   ├── 🐍 model_factory.py
│   ├── 📂 utils
│   │   ├── 📂 Vocab
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 vocab_utils.py
│   │   │   ├── 🐍 vocab.py
│   ├── 🐍 constants.py
│   ├── 🐍 data.py
│   ├── 🐍 Evaluator.py
│   ├── 🐍 preprocess_data.py
│   ├── 🐍 preprocess.py
│   ├── 🐍 tokenizer.py
│   ├── 🐍 Trainer.py
│   ├── 🐍 argparser.py
│   ├── 📄 readme.md
│   ├── 🐍 train.py
```

## Setup

While the notebooks are ready to use, script is still work in progress. 

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

# Contact

For any questions or issues regarding this repository, please contact `prasanthnaidu31k at gmail.com`
