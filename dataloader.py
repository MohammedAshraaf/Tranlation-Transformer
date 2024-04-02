from dataset import TextCustomDataset
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class TextDataModule(pl.LightningDataModule):
    def __init__(self, training_df, validation_df, transforms, special_tokens, batch_size=64):
        super().__init__()
        self.training_df = training_df
        self.validation_df = validation_df
        self.transforms = transforms
        self.special_tokens = special_tokens
        self.batch_size = batch_size
        self.train_dataset = None
        self.validation_dataset = None

    def setup(self, stage):
        """
        Builds the training and validation datasets
        :param stage:
        :return:
        """
        self.train_dataset = TextCustomDataset(
            df=self.training_df,
            split='training',
            transforms=self.transforms,
            special_tokens=self.special_tokens,

        )
        self.validation_dataset = TextCustomDataset(
            df=self.validation_df,
            split='validation',
            transforms=self.transforms,
            special_tokens=self.special_tokens,
        )

    def collate_fn(self, batch):
        """
        Prepares the current batch
        :param batch: list of records each contains x, y
        :return: the updated batch in format of x, y
        """
        src_batch, target_batch = [], []
        for src_sample, target_sample in batch:
            src_batch.append(src_sample)
            target_batch.append(target_sample)

        src_batch = pad_sequence(src_batch, padding_value=self.special_tokens['<pad>'])
        target_batch = pad_sequence(target_batch, padding_value=self.special_tokens['<pad>'])
        return src_batch, target_batch

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)