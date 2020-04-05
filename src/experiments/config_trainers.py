from src.experiments.evaluation import evaluate_classification
from src.elm.elm_trainer import ExtremeLearningAlgorithm
from src.elm.base_elm import ELM, base_extractor
from src.elm.lrf_elm import LocalReceptiveField
import time
from torch import nn
import torch
from torch import optim
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import tqdm


class ModelConfigTrainer:

    def __init__(self, dataset_name, in_dim, out_dim, name='base'):
        self.name = name
        self.dataset_name = dataset_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        self._model = None

    def build_model(self, device='cpu'):
        raise NotImplementedError

    def train_model_measure_time(self, X, y):
        raise NotImplementedError

    def config_dict(self):
        return {k: v
                for k, v in vars(self).items()
                if not k.startswith('_')}


class ElmConfigTrainer(ModelConfigTrainer):

    def __init__(self, dataset_name, in_dim, out_dim, hidden_dim,
                 activation_fn, C, try_pinverse=False):
        super().__init__(dataset_name, in_dim, out_dim, 'ELM')
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        self.C = C
        self.try_pinverse=try_pinverse

        self._model: ELM = None
        self._ela: ExtremeLearningAlgorithm = None

    def __getitem__(self, key):
        return vars(self)[key]

    def build_elm(self):
        return ELM(base_extractor(in_dim=self.in_dim,
                                  hidden_dim=self.hidden_dim,
                                  activation_fn=self.activation_fn),
                   nb_classes=self.out_dim)

    def build_ela(self):
        return ExtremeLearningAlgorithm(C=self.C)

    def build_model(self, device='cpu'):
        self._model = self.build_elm().to(device)
        self._ela = self.build_ela()

    def set_validation_data(self, X_val, y_val):
        pass

    def train_model_measure_time(self, X, y):
        if self._model is None or self._ela is None:
            self.build_model()

        tic = time.time()
        self._ela.train(self._model, X, y,
                        try_pinverse=self.try_pinverse)
        toc = time.time()

        return self._model, (toc - tic)


class LrfConfigTrainer(ModelConfigTrainer):

    def __init__(self, dataset_name, in_channels, out_dim, out_channels,
                 conv_kernel_size, pool_size,
                 C, try_pinverse=False):
        super().__init__(dataset_name, in_channels, out_dim, 'ELM')
        self.conv_kernel_size = conv_kernel_size
        self.out_channels = out_channels
        self.pool_size = pool_size
        self.C = C
        self.try_pinverse = try_pinverse

        self._model: ELM = None
        self._ela: ExtremeLearningAlgorithm = None

    def __getitem__(self, key):
        return vars(self)[key]

    def build_elm(self):
        return ELM(LocalReceptiveField(self.in_dim, self.out_channels,
                                       conv_kernel_size=self.conv_kernel_size,
                                       pool_kernel_size=self.pool_size),
                   nb_classes=self.out_dim)

    def build_ela(self):
        return ExtremeLearningAlgorithm(C=self.C)

    def build_model(self, device='cpu'):
        self._model = self.build_elm().to(device)
        self._model.H.orthogonalise_kernels()
        self._ela = self.build_ela()

    def set_validation_data(self, X_val, y_val):
        pass

    def train_model_measure_time(self, X, y):
        if self._model is None or self._ela is None:
            self.build_model()

        tic = time.time()
        self._ela.train(self._model, X, y,
                        try_pinverse=self.try_pinverse)
        toc = time.time()

        return self._model, (toc - tic)


class MLPConfigTrainer(ModelConfigTrainer):

    def __init__(self, dataset_name,
                 in_dim, out_dim, hidden_dim,
                 activation_fn, C,
                 loss_fn, batch_size,
                 optimiser, learning_rate,
                 max_epochs, patience):
        super().__init__(dataset_name, in_dim, out_dim, 'MLP')
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        self.C = C
        self.loss_fn = loss_fn()
        self.batch = batch_size
        self.lr = learning_rate
        self.optimiser = optimiser
        self.patience = patience
        self.epochs = max_epochs

        self._model: ELM = None
        self._X_val = None
        self._y_val = None
        self._device: str = 'cpu'

    def __getitem__(self, key):
        return vars(self)[key]

    def set_validation_data(self, X_val, y_val):
        self._X_val = X_val
        self._y_val = y_val

    def build_model(self, device='cpu'):
        self._model = ELM(base_extractor(in_dim=self.in_dim,
                                         hidden_dim=self.hidden_dim,
                                         activation_fn=self.activation_fn),
                          nb_classes=self.out_dim)
        self._model.initialise_out_layer(self.hidden_dim, bias=True)
        self._model = self._model.to(device)
        self._device = device

    def train_model_measure_time(self, X, y):
        if self._model is None:
            self.build_model()

        dataset = TensorDataset(X.to(self._device), y)
        dataloader = DataLoader(dataset, self.batch, shuffle=True)
        optimiser = self.optimiser(params=self._model.parameters(),
                                   lr=self.lr, weight_decay=self.C)
        early_stopper = EarlyStopping(self.patience,
                                      lambda out, y: -self.loss_fn(out, y),
                                      self._X_val, self._y_val)

        tic = time.time()

        for ep in range(self.epochs):
            for inputs, targets in dataloader:
                optimiser.zero_grad()

                # forward + backward + optimize
                response = self._model(inputs)
                loss = self.loss_fn(response, targets)
                loss.backward()
                optimiser.step()

            if early_stopper.should_stop(self._model):
                break
        toc = time.time()

        return self._model, (toc - tic)


class CNNConfigTrainer(ModelConfigTrainer):

    def __init__(self, dataset_name,
                 in_channels, out_dim, out_channels,
                 conv_kernel_size, pool_size,
                 C, loss_fn, batch_size,
                 optimiser, learning_rate,
                 max_epochs, patience, device='cpu', verbose=False):
        super().__init__(dataset_name, in_channels, out_dim, 'CNN')
        self.conv_kernel_size = conv_kernel_size
        self.out_channels = out_channels
        self.pool_size = pool_size
        self.C = C
        self.loss_fn = loss_fn()
        self.batch = batch_size
        self.lr = learning_rate
        self.optimiser = optimiser
        self.patience = patience
        self.epochs = max_epochs

        self._model: ELM = None
        self._X_val = None
        self._y_val = None
        self._device: str = device
        self._verbose: bool = verbose

    def build_model(self, device='cpu'):
        self._model = ELM(LocalReceptiveField(self.in_dim,
                                              self.out_channels,
                                              conv_kernel_size=self.conv_kernel_size,
                                              pool_kernel_size=self.pool_size),
                          nb_classes=self.out_dim).to(device)

        self._model = self._model.to(device)
        self._device = device

    def train_model_measure_time(self, X, y):
        if self._model is None:
            self.build_model(self._device)

        with torch.no_grad(): # for initialisation of last layer
            _ = self._model.forward(X[:1].to(self._device))

        dataset = TensorDataset(X.to(self._device), y)
        dataloader = DataLoader(dataset, self.batch, shuffle=True)
        optimiser = self.optimiser(params=self._model.parameters(),
                                   lr=self.lr, weight_decay=self.C)
        early_stopper = EarlyStopping(self.patience,
                                      lambda out, y: -self.loss_fn(out, y),
                                      self._X_val, self._y_val)

        tic = time.time()
        pbar = tqdm.tqdm_notebook(range(self.epochs), disable=not self._verbose)
        for ep in pbar:
            for inputs, targets in dataloader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                optimiser.zero_grad()

                # forward + backward + optimize
                response = self._model(inputs)
                loss = self.loss_fn(response, targets)
                loss.backward()
                optimiser.step()
                pbar.set_postfix({"epoch": ep, "loss": loss.detach().cpu().item()})

            if early_stopper.should_stop(self._model):
                break
        toc = time.time()

        return self._model, (toc - tic)

    def set_validation_data(self, X_val, y_val):
        self._X_val = X_val.to(self._device)
        self._y_val = y_val.to(self._device)


class EarlyStopping:

    def __init__(self, patience, score_function, X_val, y_val):
        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        self.score_function = score_function
        self.patience = patience
        self.X_val = X_val
        self.y_val = y_val
        self.counter = 0
        self.best_score = None

    def should_stop(self, model: nn.Module):
        resp = model(self.X_val)
        score = self.score_function(resp, self.y_val)

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False
