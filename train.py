from preprocessing import Preprocessing
from vocab_builder import VocabBuilder
from transformer import build_model
from torchtext.data.utils import get_tokenizer
import torch
import numpy as np
import lightning.pytorch as pl
from config import *
from training_module import TrainingModule
from dataloader import TextDataModule
from loss import CustomLoss
from callbacks import get_callbacks
import pandas as pd

# Seeds
RANDOM_STATE_SEED = 15
torch.manual_seed(RANDOM_STATE_SEED)
np.random.seed(RANDOM_STATE_SEED)
pl.seed_everything(RANDOM_STATE_SEED, workers=True)
NUMPY_RANDOM_STATE = np.random.RandomState


def main():
    # preprocess the df
    df = pd.read_csv('data/en-fr.csv', on_bad_lines='skip', nrows=100000)

    preprocessing_cls = Preprocessing()
    df = preprocess_df(df, preprocessing_cls)

    transforms = build_data_transforms(df)

    training_df, validation_df, testing_df = preprocessing_cls.split_df(df, 0.3, RANDOM_STATE_SEED)

    # create the Datasets
    dataloader_module = TextDataModule(
        training_df=training_df,
        validation_df=validation_df,
        transforms=transforms,
        special_tokens=Config.special_tokens,
        batch_size=Config.training_config['batch_size']
    )

    # create the model
    transformer = build_model(Config.model_config)

    loss_cls = CustomLoss(padding_idx=Config.special_tokens['<pad>'])

    # Training Module
    pl_training_module = TrainingModule(transformer, loss_cls, padding_idx=Config.special_tokens['<pad>'])
    callbacks = get_callbacks()

    # Training Parameters
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        benchmark=True,
        callbacks=callbacks,
        max_epochs=200,
        log_every_n_steps=100,
        limit_train_batches=100,
        limit_val_batches=500,
        accumulate_grad_batches=4,

    )

    # Training loop
    trainer.fit(pl_training_module, dataloader_module)


def preprocess_df(df, preprocessing_cls):
    df = preprocessing_cls.preprocess_df(df, ['en', 'fr'])
    df = preprocessing_cls.filter_data_by_length(df, 'en', min_length=3, max_length=25)
    return df


def build_data_transforms(df):
    token_transform = {
        'en': get_tokenizer('spacy', language='en_core_web_sm'),
        'fr': get_tokenizer('spacy', language='fr_core_news_sm'),
    }

    vocab_builder_cls = VocabBuilder(Config.special_tokens, token_transform)
    vocab_transform = {
        'en': vocab_builder_cls.build_vocab(df=df, language='en', min_freq=1),
        'fr': vocab_builder_cls.build_vocab(df=df, language='fr', min_freq=1),
    }
    transforms = [token_transform, vocab_transform]
    Config.model_config['SRC_VOCAB_SIZE'] = len(vocab_transform['en'])
    Config.model_config['TGT_VOCAB_SIZE'] = len(vocab_transform['fr'])
    return transforms


main()
