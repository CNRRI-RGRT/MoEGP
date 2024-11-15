from scipy import stats
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             mean_squared_error, mean_absolute_error, r2_score)


def evaluator(y_true, y_pred, metric='rmse'):
    if metric == 'rmse':
        return mean_squared_error(y_true, y_pred, squared=False)
    elif metric == 'mse':
        return mean_squared_error(y_true, y_pred, squared=True)
    elif metric == 'mae':
        return mean_absolute_error(y_true, y_pred)
    elif metric == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric == 'precision':
        return precision_score(y_true, y_pred, average='macro')
    elif metric == 'recall':
        return recall_score(y_true, y_pred, average='macro')
    elif metric == 'f1':
        return f1_score(y_true, y_pred, average='macro')
    elif metric == 'roc_auc':
        return roc_auc_score(y_true, y_pred)
    elif metric == 'pearson':
        result = stats.pearsonr(y_true, y_pred)[0]
        if isinstance(result, float):
            return result
        return result[0]
    elif metric == 'R2':
        return r2_score(y_true, y_pred)
    else:
        raise ValueError('Invalid metric')
