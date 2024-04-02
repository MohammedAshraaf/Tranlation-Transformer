from torch.utils.data import Dataset
import torch


class TextCustomDataset(Dataset):
    def __init__(self, df, split, transforms, special_tokens ):
        self.df = df
        self.split = split
        self.special_tokens = special_tokens
        self.transforms = transforms[:]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.iloc[idx]['en']
        y = self.df.iloc[idx]['fr']
        for transform in self.transforms:
            x = transform['en'](x)
            y = transform['fr'](y)
        x = self.tensor_transform(x)
        y = self.tensor_transform(y)
        return x, y

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(self, token_ids):
        return torch.cat((torch.tensor([self.special_tokens['<bos>']]),
                          torch.tensor(token_ids),
                          torch.tensor([self.special_tokens['<eos>']])))

