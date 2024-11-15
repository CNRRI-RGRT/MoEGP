import argparse
import os.path
import warnings
from itertools import product

import pandas as pd
import torch
from torch import nn

from utils.evaluator import evaluator
from data_loader import MyDataLoader
from trait_training import Trainer
from utils.task_model import TaskModel
from utils.moe import Gating, creat_mlp_layer

warnings.filterwarnings("ignore")
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def get_parse():
    parser = argparse.ArgumentParser(description='genomics prediction based on MoE, only support regression task type')
    parser.add_argument('--input_dir', type=str, default='data/processed')
    parser.add_argument('--input_json', type=str, default='input_data.json')
    parser.add_argument('--output_dir', type=str, default='model/MoEGP')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', nargs='+', default=[1e-4])
    parser.add_argument('--dropouts', nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--hidden_dim_list', nargs='+', default=[[512], [128], [1024, 512]])
    parser.add_argument('--num_experts', nargs='+', default=[6, 10, 20])
    parser.add_argument('--best_metrics', type=str, default='pearson', choices=['pearson', 'rmse'],
                        help='select the best model through metrics, default pearson')
    parser.add_argument('--zscore', type=bool, default=True, help='whether to use zscore normalization in the modeling')
    return parser.parse_args()


def convert_hidden_str_to_list(hidden_str):
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
    batch_size = args.batch_size
    best_metrics = args.best_metrics
    if isinstance(args.hidden_dim_list, list):
        hidden_dim_list = args.hidden_dim_list
    else:
        hidden_dim_list = [list(map(int, x)) for x in convert_hidden_str_to_list(args.hidden_dim_list)]
    num_experts = list(map(int, args.num_experts))
    lrs = list(map(float, args.lr))
    dropouts = list(map(float, args.dropouts))
    zscore = args.zscore
    result = [['Name', "rmse", "pearson", "mse", "mae", "R2", 'DropOut', 'Lr', 'Hidden Dim', 'Num Expert']]

    for task in TaskModel.load(input_json):
        data = MyDataLoader(trait=os.path.join(input_dir, task.data_dir),
                            genotype_file=os.path.join(input_dir, task.genotypes_dir),
                            device=device, zscore=zscore).load_2d()

        print(f'{task.name} Training')
        os.makedirs(os.path.join(output_dir, task.name), exist_ok=True)

        for idx, (lr, dropout, hidden_dim, num_expert) in enumerate(
                product(lrs, dropouts, hidden_dim_list, num_experts)):
            print(f'lr: {lr}, dropout:{dropout}, hidden_dim: {hidden_dim}, num_expert: {num_expert}')
            loss_fn = nn.MSELoss()
            init_model = creat_mlp_layer(task.input_dim, task.out_dim, hidden_dim, dropout=dropout,
                                         batch_norm=True).to(device)
            model = Gating(model=init_model, num_experts=num_expert,
                           input_dim=task.input_dim, loss_coef=1, top_k=2).to(device)
            trainer = Trainer(data=data, device=device, epochs=epochs, batch_size=batch_size, lr=lr,
                              task_model=task, loss_fn=loss_fn)
            model_rmse, model_pearson, best_metrics_rmse, best_metrics_pearson = trainer.train(model)  # RMSE

            if best_metrics == 'RMSE':
                init_best_metrics = float('+inf')  # RMSE
                val_y_true_best_rmse, val_pred_best_rmse, _ = trainer.validate(model_rmse)
                metrics_scores = []
                for metric in task.metrics_names:
                    score = evaluator(val_y_true_best_rmse, val_pred_best_rmse, metric)
                    print(f'RMSE MODEL Valid {metric}: {score}')
                    metrics_scores.append(score)
                if best_metrics_rmse < init_best_metrics:
                    temp_result = [task.name] + metrics_scores + [dropout, lr, hidden_dim, num_expert]
                    init_best_metrics = best_metrics_rmse
                    y_pred = val_pred_best_rmse

            elif best_metrics == 'pearson':
                init_best_metrics = float('-inf')  # Pearson
                val_y_true_best_pearson, val_pred_best_pearson, _ = trainer.validate(model_pearson)
                metrics_scores = []
                for metric in task.metrics_names:
                    score = evaluator(val_y_true_best_pearson, val_pred_best_pearson, metric)
                    print(f'Pearson MODEL Valid {metric}: {score}')
                    metrics_scores.append(score)
                if best_metrics_pearson > init_best_metrics:
                    temp_result = [task.name] + metrics_scores + [dropout, lr, hidden_dim, num_expert]
                    init_best_metrics = best_metrics_pearson
                    y_pred = val_pred_best_pearson
            else:
                raise ValueError('best_metrics is not valid, only support rmse and pearson')

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
        test_data = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true})
        test_data.to_csv(os.path.join(output_dir, task.name, f'{task.name}_Pred.csv'), index=False)

        result.append(temp_result)
        df = pd.DataFrame(result)
        df.to_csv(os.path.join(output_dir, f'metrics.csv'))
        torch.cuda.empty_cache()


main()
