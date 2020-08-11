from src.experiments.evaluation import Evaluator
from src.experiments.config_trainers import ModelConfigTrainer, ElmConfigTrainer, LrfConfigTrainer
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Type
from sklearn.model_selection import ParameterGrid
from src.experiments.single_experiment import repeated_experiment
import pandas as pd
import tqdm


class GridSearchExperiment:

    def __init__(self, dataset_name: str,
                 model_name: str,
                 trainer_class: Type,
                 val_evaluator: Evaluator,
                 searched_params: Dict[str, List],
                 constant_params: Dict,
                 repetitions: int = 1, 
                 return_model: bool = False,
                 verbose: bool = False,
                 custom_tqdm = tqdm.tqdm):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.parameter_grid = ParameterGrid(searched_params)
        self.constant_params = constant_params
        self.trainer_class = trainer_class
        self.evaluator = val_evaluator
        self.repetitions = repetitions
        self.return_model = return_model
        self.verbose = verbose
        self.tqdm = custom_tqdm
        self._results_cache = []

    def run(self, train_dataset: Dataset, val_dataset: Dataset):
        results = []
        for setup in self.tqdm(self.parameter_grid, disable=not self.verbose):
            if  self.trainer_class in (ElmConfigTrainer,  LrfConfigTrainer):
                setup_to_use = {k: setup[k]
                                    for k in setup if k != "batch_size"}
            else:
                setup_to_use = {k: setup[k] for k in setup}
                
            trainer = self.trainer_class(self.dataset_name,
                                         **setup_to_use,
                                         **self.constant_params)
            if hasattr(trainer, "set_validation_data"):
                trainer.set_validation_data(dataset_val=val_dataset)
            results += repeated_experiment(trainer, self.evaluator,
                                           data_loader=DataLoader(train_dataset,
                                                                  batch_size=setup["batch_size"],
                                                                  num_workers=1),
                                           repetitions=self.repetitions,
                                           experiment_metadata={"model": self.model_name,
                                                                **setup},
                                           return_model=self.return_model)
            pd.DataFrame(results).to_pickle("experiments/gridsearch/backup_results.pkl")
        return pd.DataFrame(results)


if __name__ == "__main__":
    print("hello world")
    import torch
    import torch.nn as nn
    from src.elm.base_elm import ELM, base_extractor
    from src.experiments.config_trainers import ElmConfigTrainer
    from src.experiments.evaluation import ClassificationDataLoaderEvaluator
    from torch.utils.data import TensorDataset
    dataset = TensorDataset(torch.randn(
        100, 10), torch.randint(2, (100,)).long())
    params = {"C": [0.1, 0.01], "activation_fn": [nn.ReLU, nn.Sigmoid],
              "hidden_dim": [10, 20], "batch_size": [16, 32]}
    consts = {"in_dim": 10, "out_dim": 2, "try_pinverse": True}
    evaluator = ClassificationDataLoaderEvaluator(
        DataLoader(dataset, batch_size=100))
    gse = GridSearchExperiment(
        "test_data", "elm", ElmConfigTrainer, evaluator, params, consts, 5)
    res = gse.run(dataset, dataset)
    print(res)
