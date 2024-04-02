**Machine Translation Using Transformers (PyTorch)**

## Overview

This repository contains an implementation of machine translation using transformers in PyTorch. The main objective is to translate text from English to French utilizing the transformer architecture.

## Dataset

The dataset used for training and evaluation is available at [Kaggle](https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset). It consists of English-French parallel texts, which are essential for training a translation model.

## Requirements

To run the code in this repository, ensure you install the required Python packages via pip:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Download the dataset from the provided link and place the csv in the `data/` directory.

2. **Training**: Run the training script to train the transformer model on the English-French translation task. Adjust the configurations in the `config.py` as needed.

    ```bash
    python train.py
    ```

## Model Architecture

The translation model utilizes transformer architecture, which is a deep learning model introduced in the paper "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" by Vaswani et al. Transformers have shown remarkable performance in various natural language processing tasks, including machine translation.