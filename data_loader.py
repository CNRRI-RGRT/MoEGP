from dataclasses import dataclass

import torch
import pandas as pd
import numpy as np
from torch.utils.data import dataset, Dataset


__all__ = ['MyDataLoader', 'MyDataset']


@dataclass
class MyDataset:
    train_data: Dataset
    valid_data: Dataset
    mean: float
    std_dev: float


@dataclass
class MyDataLoader:
    trait: str
    genotype_file: str
    device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_channel: int = 3
    resize: int = 0
    zscore: bool = True

    def load_2d(self):
        df = pd.read_csv(self.trait, index_col=0)
        mean = np.mean(df.iloc[:, 0], axis=0)
        std_dev = np.std(df.iloc[:, 0], axis=0)
        if self.zscore:
            df['normalization'] = (df.iloc[:, 0] - mean) / std_dev

        genotype_df = pd.read_csv(self.genotype_file, index_col=0).T
        genotype = genotype_df.to_dict('list')
        train_df = df[df['group'] == 'training']
        valid_df = df[df['group'] == 'validation']

        train_x = torch.FloatTensor(np.array([genotype[y] for y in train_df.index.tolist()])).to(device=self.device)
        valid_x = torch.FloatTensor(np.array([genotype[y] for y in valid_df.index.tolist()])).to(device=self.device)

        # y
        if self.zscore:
            train_y = torch.FloatTensor(train_df['normalization'].values.tolist()).to(self.device).reshape(len(train_df), -1)
            valid_y = torch.FloatTensor(valid_df['normalization'].values.tolist()).to(self.device).reshape(len(valid_df), -1)
        else:
            train_y = torch.FloatTensor(train_df.iloc[:, -2].values.tolist()).to(self.device).reshape(len(train_df), -1)
            valid_y = torch.FloatTensor(valid_df.iloc[:, -2].values.tolist()).to(self.device).reshape(len(valid_df), -1)

        train_dataset = dataset.TensorDataset(train_x, train_y)
        valid_dataset = dataset.TensorDataset(valid_x, valid_y)
        return MyDataset(train_dataset, valid_dataset, mean, std_dev)
