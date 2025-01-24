import argparse
import time
import os.path
import warnings
import copy
from itertools import product

import pandas as pd
import numpy as np
import torch
from torch import nn

from utils.evaluator import evaluator
from data_loader import MyDataLoader
from trait_training import Trainer
from utils.task_model import TaskModel
from utils.moe import Gating, creat_mlp_layer

warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_parse():
    parser = argparse.ArgumentParser(description='genomics prediction based on MoE, only support regression task type')
    parser.add_argument('--input_dir', type=str, default='data/processed')
    parser.add_argument('--input_json', type=str, default='input_data_sample.json')   # 8
    parser.add_argument('--output_dir', type=str, default='model/marker_compare_rmse_10')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)  # ATI is 16
    parser.add_argument('--cross_fold', type=int, default=10)
    parser.add_argument('--lr', nargs='+', default=[1e-4, 1e-5])  # [1e-4, 1e-5]
    parser.add_argument('--dropouts', nargs='+', default=[0.3, 0.5])  # [0.3, 0.5]
    parser.add_argument('--num_experts', nargs='+', default=[6, 10, 20])  # [6, 10]
    parser.add_argument('--best_metrics', type=str, default='rmse', choices=['pearson', 'rmse'],
                        help='select the best model through metrics, default pearson')
    parser.add_argument('--zscore', type=bool, default=True, help='whether to use zscore normalization in the modeling')
    return parser.parse_args()


def convert_hidden_input_to_list(hidden_str):
    a_str = ''
    for i in hidden_str:
        a_str += i.replace("'", '')
    b_list = []
    for i in a_str.split(']['):
        b_list.append(i.replace("'", '').replace('[', '').replace(']', '').split(','))
    return b_list


def main():
    args = get_parse()
    input_dir = args.input_dir
    input_json = args.input_json
    output_dir = args.output_dir
    epochs = args.epochs
    # batch_size = args.batch_size
    best_metrics = args.best_metrics
    cross_fold = args.cross_fold
    num_experts = list(map(int, args.num_experts))
    lrs = list(map(float, args.lr))
    dropouts = list(map(float, args.dropouts))
    zscore = args.zscore

    for task in TaskModel.load(input_json):
        result = [['Name', "rmse", "pearson", "mse", "mae", "R2", 'DropOut', 'Lr', 'Hidden Dim', 'Num Expert']]
        result_ave = [['Name', "rmse", "pearson", "mse", "mae", "R2"]]
        time_result = []
        for trait in task.traits:

            load_data = MyDataLoader(trait=os.path.join(input_dir, task.data_dir, f'{trait}_processed.csv'),
                                     genotype_file=os.path.join(input_dir, task.data_dir, task.genotypes_dir),
                                     device=device, zscore=zscore, k_fold=cross_fold)
            print(f'{task.name} {trait} Training')
            os.makedirs(os.path.join(output_dir, task.name), exist_ok=True)

            score_list = []
            for cross_idx, data in enumerate(load_data.load_2d()):
                print(f'data cross validation: {cross_idx}')
                if best_metrics.upper() == 'RMSE':
                    init_best_metrics = float('+inf')  # RMSE
                elif best_metrics.upper() == 'PEARSON':
                    init_best_metrics = float('-inf')  # Pearson
                else:
                    raise ValueError(f'Best metric {best_metrics} not supported')

                trait_time = []
                best_model = None
                for idx, (lr, dropout, hidden_dim, num_expert) in enumerate(
                        product(lrs, dropouts, task.hidden_size, num_experts)):
                    torch.cuda.empty_cache()
                    start = time.time()
                    print(f'{idx} lr: {lr}, dropout:{dropout}, hidden_dim: {hidden_dim}, num_expert: {num_expert}')
                    loss_fn = nn.MSELoss()
                    init_model = creat_mlp_layer(task.input_dim, task.out_dim, hidden_dim, dropout=dropout,
                                                 batch_norm=True).to(device)
                    model = Gating(model=init_model, num_experts=num_expert,
                                   input_dim=task.input_dim, loss_coef=0.1, top_k=2).to(device)
                    trainer = Trainer(data=data, device=device, epochs=epochs, batch_size=task.batch_size, lr=lr,
                                      task_model=task, loss_fn=loss_fn)
                    model_rmse, model_pearson, best_metrics_rmse, best_metrics_pearson = trainer.train(model)  # RMSE

                    if best_metrics.upper() == 'RMSE':
                        val_y_true_best_rmse, val_pred_best_rmse, _ = trainer.validate(model_rmse)
                        metrics_scores = []
                        for metric in task.metrics_names:
                            score = evaluator(val_y_true_best_rmse, val_pred_best_rmse, metric)
                            print(f'RMSE MODEL Valid {metric}: {score}')
                            metrics_scores.append(score)
                        if best_metrics_rmse < init_best_metrics:
                            temp_result = [task.name, trait] + metrics_scores + [dropout, lr, hidden_dim, num_expert]
                            init_best_metrics = best_metrics_rmse
                            y_pred = val_pred_best_rmse
                            best_model = copy.deepcopy(model_rmse)

                    elif best_metrics.upper() == 'PEARSON':
                        val_y_true_best_pearson, val_pred_best_pearson, _ = trainer.validate(model_pearson)
                        metrics_scores = []
                        for metric in task.metrics_names:
                            score = evaluator(val_y_true_best_pearson, val_pred_best_pearson, metric)
                            print(f'Pearson MODEL Valid {metric}: {score}')
                            metrics_scores.append(score)
                        if best_metrics_pearson > init_best_metrics:
                            temp_result = [task.name, trait] + metrics_scores + [dropout, lr, hidden_dim, num_expert]
                            init_best_metrics = best_metrics_pearson
                            y_pred = val_pred_best_pearson
                            best_model = copy.deepcopy(model_pearson)
                    else:
                        raise ValueError('best_metrics is not valid, only support rmse and pearson')
                    use_time = time.time() - start
                    trait_time.append(use_time)

                    del model_rmse, model_pearson, model, init_model

                #  rmse
                torch.save(best_model, os.path.join(output_dir, task.name, f'{trait}_model_{round(init_best_metrics, 5)}.pt'))

                test_y_true = []
                for _, y in data.valid_data:
                    test_y_true.append(y.cpu().numpy()[0])

                # de normalization
                if zscore:
                    y_true = [y * data.std_dev + data.mean for y in test_y_true]
                    y_pred = [x[0] * data.std_dev + data.mean for x in y_pred]
                else:
                    y_true = test_y_true
                    y_pred = [x[0] for x in y_pred]

                best_score = []
                for metric in task.metrics_names:
                    score = evaluator(y_true, y_pred, metric)
                    best_score.append(score)
                score_list.append(best_score)
                test_data = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true})
                test_data.to_csv(os.path.join(output_dir, task.name, f'{trait}_Pred_Cross{cross_idx}.csv'), index=False)

                # save
                result.append([task.name, trait] + best_score)
                df = pd.DataFrame(result)
                df.to_csv(os.path.join(output_dir, f'MoEGP_{task.name}_metrics_cross_validation2.csv'))

            result_ave.append([task.name, trait, 'training'] + list(np.average(np.array(score_list), axis=0)))
            df = pd.DataFrame(result_ave)
            df.to_csv(os.path.join(output_dir, f'MoEGP_{task.name}_metrics_ave2.csv'))


if __name__ == '__main__':
    main()
