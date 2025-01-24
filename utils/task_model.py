import json
from enum import Enum
from dataclasses import dataclass


__all__ = ['TaskType', 'TaskModel']

from typing import List


class TaskType(Enum):
    REGRESSION = 0
    CLASSIFICATION = 1

    @staticmethod
    def get_type(task_type) -> "TaskType":
        if task_type == 'regression':
            return TaskType.REGRESSION
        return TaskType.CLASSIFICATION


@dataclass(frozen=True)
class TaskModel:
    data_dir: str
    traits: List[str]
    name: str
    mode: str
    metrics_names: List[str]
    out_dim: int
    genotypes_dir: str
    input_dim: int
    hidden_size: List[int]
    batch_size: int

    @staticmethod
    def load(file) -> List["TaskModel"]:
        with open(file, 'r') as f:
            data = json.load(f)
        for name, task in data.items():
            yield TaskModel(
                data_dir=task['data_dir'],
                name=name,
                mode=task['mode'],
                metrics_names=task['metric_name'],
                out_dim=task['out_dim'],
                genotypes_dir=task['genotypes_dir'],
                input_dim=task['input_dim'],
                traits=task['traits'],
                batch_size=task['batch_size'],
                hidden_size=task.get('hidden_size', [[512], [1024, 512]]),
            )
