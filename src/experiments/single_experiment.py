from src.experiments.evaluation import Evaluator
from src.experiments.config_trainers import ModelConfigTrainer
import numpy as np
from torch.utils.data import DataLoader

def single_experiment(model_config_trainer: ModelConfigTrainer,
                      evaluator: Evaluator,
                      X: np.ndarray = None, y: np.ndarray = None,
                      experiment_metadata=None,
                      device='cpu',
                      return_model=False,
                      data_loader: DataLoader=None):
    if experiment_metadata is None:
        experiment_metadata = dict()

    model_config_trainer.build_model(device)
    error = True

    while error:
        if data_loader is not None:
            model, time, error = model_config_trainer.train_with_dataloader_measure_time(data_loader)
        else:
            model, time, error = model_config_trainer.train_model_measure_time(X, y)
    return {**experiment_metadata,
            **model_config_trainer.config_dict(),
            **evaluator.evaluate(model),
            'time': time,
            'error': error,
            'model': model if return_model else None}


def repeated_experiment(model_config_trainer: ModelConfigTrainer,
                        evaluator: Evaluator,
                        X: np.ndarray = None, y: np.ndarray = None,
                        repetitions: int = 2,
                        experiment_metadata=None,
                        device='cpu',
                        data_loader: DataLoader = None,
                        return_model=False):
    if experiment_metadata is None:
        experiment_metadata = dict()

    results = []
    for i in range(repetitions):
        result = single_experiment(model_config_trainer,
                                   evaluator,
                                   X, y,
                                   experiment_metadata,
                                   device, data_loader=data_loader)
        results.append({'repetition': i,
                        **result})
    return results
