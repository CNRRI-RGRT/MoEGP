from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from data_loader import MyDataset
from utils.evaluator import evaluator
from utils.task_model import TaskModel, TaskType

__all__ = ['Trainer']


@dataclass
class Trainer:
    data: MyDataset
    task_model: TaskModel
    device: torch.device
    batch_size: int = 64
    epochs: int = 100
    lr: float = 0.001
    loss_fn: Callable = nn.MSELoss()

    def plot(self, train_plot_loss: list, val_plot_loss: list):
        plt.plot(train_plot_loss[2:], label='train', color='blue')
        plt.plot(val_plot_loss[2:], label='val', color='red')
        plt.legend()
        plt.show()

    @staticmethod
    def get_best_params_rmse(model, best_state_dict: dict, best_metric: float, metric_score: float) -> Tuple[dict, float]:
        if metric_score < best_metric:
            best_metric = metric_score
            best_state_dict = deepcopy(model.state_dict())
        return best_state_dict, best_metric

    @staticmethod
    def get_best_params_pearson(model, best_state_dict: dict, best_metric: float, metric_score: float) -> Tuple[dict, float]:
        if metric_score > best_metric:
            best_metric = metric_score
            best_state_dict = deepcopy(model.state_dict())
        return best_state_dict, best_metric

    def train(self, model: nn.Module):
        train_data = DataLoader(self.data.train_data, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-6)

        best_metrics_rmse = float('+inf')  # 选取rmse 最低的
        best_metrics_pearson = float('-inf')  # 选取pearson 最高的
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

        train_plot_loss = []
        valid_plot_loss = []
        best_state_dict_rmse, best_state_dict_pearson = {}, {}

        for epoch in range(self.epochs):
            torch.cuda.empty_cache()
            model.train()
            train_y_true = []
            train_y_pred = []
            loss_list = []
            for x, y in train_data:
                optimizer.zero_grad()
                result = model(x)
                if isinstance(result, tuple):
                    output, aux_loss = result
                    loss = self.loss_fn(output, y)
                    total_loss = loss + aux_loss
                else:
                    output = result
                    total_loss = self.loss_fn(output, y)
                total_loss.backward()
                optimizer.step()
                train_y_pred.extend(output.data.tolist())
                train_y_true.extend(y.data.tolist())
                loss_list.append(total_loss.item())
            val_y_true, val_pred, val_loss = self.validate(model)
            scheduler.step(val_loss)
            train_loss = np.mean(loss_list)
            train_plot_loss.append(train_loss)
            valid_plot_loss.append(val_loss)
            valid_best_metrics_rmse = evaluator(val_y_true, val_pred, 'rmse')
            valid_best_metrics_pearson = evaluator(val_y_true, val_pred, 'pearson')

            # print
            if (epoch + 1) % 50 == 0:
                print(f'Training Epoch: {epoch}, Loss: {train_loss:.4f}, '
                      f'{self.task_model.metrics_names[0]}: '
                      f'{evaluator(train_y_true, train_y_pred, self.task_model.metrics_names[0])}, '
                      f'{self.task_model.metrics_names[1]}: '
                      f'{evaluator(train_y_true, train_y_pred, self.task_model.metrics_names[1])}')

                print(f'Valid Epoch: {epoch}, Loss: {val_loss:.4f}, '
                      f'{self.task_model.metrics_names[0]}: {valid_best_metrics_rmse: .4f}, '
                      f'{self.task_model.metrics_names[1]}: {valid_best_metrics_pearson: .4f}')

            best_state_dict_rmse, best_metrics_rmse = (
                self.get_best_params_rmse(model, best_state_dict_rmse, best_metrics_rmse, valid_best_metrics_rmse))
            best_state_dict_pearson, best_metrics_pearson = (
                self.get_best_params_pearson(model, best_state_dict_pearson, best_metrics_pearson, valid_best_metrics_pearson))

        # self.plot(train_plot_loss, valid_plot_loss)
        model_rmse = deepcopy(model)
        model_pearson = deepcopy(model)
        if best_state_dict_rmse:
            model_rmse.load_state_dict(best_state_dict_rmse)
            print(f'Best Model RMSE: {best_metrics_rmse}')

        if best_state_dict_pearson:
            model_pearson.load_state_dict(best_state_dict_pearson)
            print(f'Best Model Pearson: {best_metrics_pearson}')

        return model_rmse, model_pearson, best_metrics_rmse, best_metrics_pearson

    @torch.no_grad()
    def validate(self, val_model: nn.Module):
        val_model.eval()
        val_data = DataLoader(self.data.valid_data, batch_size=self.batch_size, shuffle=False)
        val_y_true = []
        val_pred = []
        loss_list = []
        for x, y in val_data:
            result = val_model(x)
            if isinstance(result, tuple):
                output, aux_loss = result
                loss = self.loss_fn(output, y)
                total_loss = loss + aux_loss
            else:
                output = result
                total_loss = self.loss_fn(output, y)
            val_y_true.extend(y.data.tolist())
            val_pred.extend(output.data.tolist())
            loss_list.append(total_loss.item())
        return val_y_true, val_pred, np.mean(loss_list)
