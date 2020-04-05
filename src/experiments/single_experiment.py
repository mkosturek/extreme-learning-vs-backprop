from src.experiments.evaluation import Evaluator
from src.experiments.config_trainers import ModelConfigTrainer
import numpy as np


def single_experiment(model_config_trainer: ModelConfigTrainer,
                      evaluator: Evaluator,
                      X: np.ndarray, y: np.ndarray,
                      experiment_metadata=None,
                      device='cpu',
                      return_model=False):
    if experiment_metadata is None:
        experiment_metadata = dict()

    model_config_trainer.build_model(device)
    model, time = model_config_trainer.train_model_measure_time(X, y)
    return {**experiment_metadata,
            **model_config_trainer.config_dict(),
            **evaluator.evaluate(model),
            'time': time,
            'model': model if return_model else None}


def repeated_experiment(model_config_trainer: ModelConfigTrainer,
                        evaluator: Evaluator,
                        X: np.ndarray, y: np.ndarray,
                        repetitions: int,
                        experiment_metadata=None,
                        device='cpu'):
    if experiment_metadata is None:
        experiment_metadata = dict()

    results = []
    for i in range(repetitions):
        result = single_experiment(model_config_trainer,
                                   evaluator,
                                   X, y,
                                   experiment_metadata,
                                   device)
        results.append({'repetition': i,
                        **result})
    return results
