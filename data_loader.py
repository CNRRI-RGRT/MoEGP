from dataclasses import dataclass

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
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
    zscore: bool = True
    k_fold: int = 10

    def kfold(self) -> KFold:
        return KFold(n_splits=self.k_fold, shuffle=True, random_state=100)

    def load_2d(self):
        df = pd.read_csv(self.trait, index_col=0)
        mean = np.mean(df.iloc[:, 0], axis=0)
        std_dev = np.std(df.iloc[:, 0], axis=0)
        y = np.array(df.iloc[:, 0].values.tolist())
        if self.zscore:
            df['normalization'] = (df.iloc[:, 0] - mean) / std_dev
            y = np.array(df['normalization'].values.tolist())

        genotype_df = pd.read_csv(self.genotype_file, index_col=0).T
        genotype = genotype_df.to_dict('list')
        x = np.array([genotype[i] for i in df.index.tolist()])

        for train_index, test_index in self.kfold().split(x):
            train_x, train_y = x[train_index], y[train_index]
            test_x, test_y = x[test_index], y[test_index]

            # x
            train_x = torch.FloatTensor(train_x).to(device=self.device)
            valid_x = torch.FloatTensor(test_x).to(device=self.device)

            # y
            train_y = torch.FloatTensor(train_y).to(self.device).reshape(len(train_y), -1)
            valid_y = torch.FloatTensor(test_y).to(self.device).reshape(len(test_y), -1)
            train_dataset = dataset.TensorDataset(train_x, train_y)
            valid_dataset = dataset.TensorDataset(valid_x, valid_y)
            yield MyDataset(train_dataset, valid_dataset, mean, std_dev)
