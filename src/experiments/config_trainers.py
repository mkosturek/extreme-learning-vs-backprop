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
from torch import nn


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

    def train_with_dataloader_measure_time(self, data_loader: DataLoader):
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
        self.try_pinverse = try_pinverse

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

    def set_validation_data(self, X_val=None, y_val=None, dataset_val=None):
        pass

    def train_model_measure_time(self, X, y):
        if self._model is None or self._ela is None:
            self.build_model()

        tic = time.time()
        self._ela.train(self._model, X, y,
                        try_pinverse=self.try_pinverse)
        toc = time.time()

        return self._model, (toc - tic), False

    def train_with_dataloader_measure_time(self, data_loader):
        if self._model is None or self._ela is None:
            self.build_model()

        tic = time.time()
        self._ela.train_data_loader(self._model, data_loader,
                                    try_pinverse=self.try_pinverse)
        toc = time.time()

        return self._model, (toc - tic), False


class LrfConfigTrainer(ModelConfigTrainer):

    def __init__(self, dataset_name, in_channels, out_dim, out_channels,
                 conv_kernel_size, pool_size,
                 C, pool_stride=2, try_pinverse=False):
        super().__init__(dataset_name, in_channels, out_dim, 'ELM')
        self.conv_kernel_size = conv_kernel_size
        self.out_channels = out_channels
        self.pool_size = pool_size
        self.C = C
        self.try_pinverse = try_pinverse
        self.pool_stride = pool_stride

        self._model: ELM = None
        self._ela: ExtremeLearningAlgorithm = None

    def __getitem__(self, key):
        return vars(self)[key]

    def build_elm(self):
        return ELM(LocalReceptiveField(self.in_dim, self.out_channels,
                                       conv_kernel_size=self.conv_kernel_size,
                                       pool_kernel_size=self.pool_size,
                                       pool_stride=self.pool_stride),
                   nb_classes=self.out_dim)

    def build_ela(self):
        return ExtremeLearningAlgorithm(C=self.C)

    def build_model(self, device='cpu'):
        self._model = self.build_elm().to(device)
        self._model.H.orthogonalise_kernels()
        self._ela = self.build_ela()

    def set_validation_data(self, X_val=None, y_val=None, dataset_val=None):
        pass

    def train_model_measure_time(self, X, y):
        if self._model is None or self._ela is None:
            self.build_model()

        tic = time.time()
        self._ela.train(self._model, X, y,
                        try_pinverse=self.try_pinverse)
        toc = time.time()

        return self._model, (toc - tic), False

    def train_with_dataloader_measure_time(self, data_loader):
        if self._model is None or self._ela is None:
            self.build_model()

        tic = time.time()
        self._ela.train_data_loader(self._model, data_loader,
                                    try_pinverse=self.try_pinverse)
        toc = time.time()

        return self._model, (toc - tic), False


class MLPConfigTrainer(ModelConfigTrainer):

    def __init__(self, dataset_name,
                 in_dim, out_dim, hidden_dim,
                 activation_fn, C,
                 loss_fn, batch_size,
                 optimiser, learning_rate,
                 max_epochs, patience,
                 nb_layers=1,
                 softmax=False):
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
        self.softmax = softmax

        self.nb_layers = nb_layers
        
        self._model: ELM = None
        self._dataset_val = None
        self._device: str = 'cpu'

    def __getitem__(self, key):
        return vars(self)[key]

    def set_validation_data(self, X_val=None, y_val=None, dataset_val=None):
        if X_val is not None and y_val is not None:
            self._dataset_val = TensorDataset(X_val, y_val)
        elif dataset_val is not None:
            self._dataset_val = dataset_val
        else:
            raise RuntimeError()

    def build_model(self, device='cpu'):
        if self.nb_layers <= 1:
            self._model = ELM(base_extractor(in_dim=self.in_dim,
                                            hidden_dim=self.hidden_dim,
                                            activation_fn=self.activation_fn),
                            nb_classes=self.out_dim)
        else:
            extractor = base_extractor(in_dim=self.in_dim,
                                            hidden_dim=self.hidden_dim,
                                            activation_fn=self.activation_fn)
            sequence = [extractor]
            for i in range(self.nb_layers - 1):
                sequence.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                sequence.append(self.activation_fn())
            
            extractor = nn.Sequential(*sequence)
            self._model = ELM(extractor, nb_classes=self.out_dim)
        self._model.initialise_out_layer(self.hidden_dim, bias=True, softmax=self.softmax)
        self._model = self._model.to(device)
        self._device = device

    def train_model_measure_time(self, X, y):
        if self._model is None:
            self.build_model()

        dataset = TensorDataset(X, y)
        return self._train_with_dataset(dataset)

    def train_with_dataloader_measure_time(self, data_loader):
        if self._model is None:
            self.build_model()

        dataset = data_loader.dataset
        return self._train_with_dataset(dataset)

    def _train_with_dataset(self, dataset):
        dataloader = DataLoader(dataset, self.batch, shuffle=True)
        optimiser = self.optimiser(params=self._model.parameters(),
                                   lr=self.lr, weight_decay=self.C)
        early_stopper = EarlyStopping(self.patience,
                                      lambda out, y: -self.loss_fn(out, y),
                                      self._dataset_val,
                                      self._device)

        tic = time.time()

        for ep in range(self.epochs):
            for inputs, targets in dataloader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                optimiser.zero_grad()

                # forward + backward + optimize
                response = self._model(inputs)
                loss = self.loss_fn(response, targets)
                if torch.isnan(loss):
                    return self._model, time.time() - tic, True
                loss.backward()
                optimiser.step()

            if early_stopper.should_stop(self._model):
                break
        toc = time.time()

        return self._model, (toc - tic), False


class CNNConfigTrainer(ModelConfigTrainer):

    def __init__(self, dataset_name,
                 in_channels, out_dim, out_channels,
                 conv_kernel_size, pool_size,
                 C, loss_fn, batch_size,
                 optimiser, learning_rate,
                 max_epochs, patience, pool_stride=2,
                 device='cpu', verbose=False):
        super().__init__(dataset_name, in_channels, out_dim, 'CNN')
        self.conv_kernel_size = conv_kernel_size
        self.out_channels = out_channels
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.C = C
        self.loss_fn = loss_fn()
        self.batch = batch_size
        self.lr = learning_rate
        self.optimiser = optimiser
        self.patience = patience
        self.epochs = max_epochs

        self._model: ELM = None
        self._dataset_val = None

        self._device: str = device
        self._verbose: bool = verbose

    def build_model(self, device='cpu'):
        self._model = ELM(LocalReceptiveField(self.in_dim,
                                              self.out_channels,
                                              conv_kernel_size=self.conv_kernel_size,
                                              pool_kernel_size=self.pool_size,
                                              pool_stride=self.pool_stride),
                          nb_classes=self.out_dim).to(device)

        self._model = self._model.to(device)
        self._device = device

    def train_model_measure_time(self, X, y):
        if self._model is None:
            self.build_model(self._device)

        with torch.no_grad():  # for initialisation of last layer
            _ = self._model.forward(X[:1].to(self._device))

        dataset = TensorDataset(X, y)
        return self._train_with_dataset(dataset)

    def train_with_dataloader_measure_time(self, data_loader):
        if self._model is None:
            self.build_model(self._device)
        return self._train_with_dataset(data_loader.dataset)

    def _train_with_dataset(self, dataset):
        dataloader = DataLoader(dataset, self.batch,
                                shuffle=True, num_workers=1)
        optimiser = self.optimiser(params=self._model.parameters(),
                                   lr=self.lr, weight_decay=self.C)
        early_stopper = EarlyStopping(self.patience,
                                      lambda out, y: -self.loss_fn(out, y),
                                      self._dataset_val,
                                      self._device)

        tic = time.time()
        pbar = tqdm.tqdm_notebook(
            range(self.epochs), disable=not self._verbose)
        for ep in pbar:
            for inputs, targets in dataloader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                optimiser.zero_grad()

                # forward + backward + optimize
                response = self._model(inputs)
                loss = self.loss_fn(response, targets)
                if torch.isnan(loss):
                    return self._model, time.time() - tic, True
                loss.backward()
                optimiser.step()
            pbar.set_postfix({"epoch": ep, "loss": loss.detach().cpu().item()})

            if early_stopper.should_stop(self._model):
                break
        toc = time.time()

        return self._model, (toc - tic), False

    def set_validation_data(self, X_val=None, y_val=None, dataset_val=None):
        if X_val is not None and y_val is not None:
            self._dataset_val = TensorDataset(X_val, y_val)
        elif dataset_val is not None:
            self._dataset_val = dataset_val
        else:
            raise RuntimeError()


class EarlyStopping:

    def __init__(self, patience, score_function, dataset=None, device='cpu'):
        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        self.score_function = score_function
        self.patience = patience
        self.data_loader = DataLoader(dataset, batch_size=2048)
        self.counter = 0
        self.best_score = None
        self.device = device

    @torch.no_grad()
    def should_stop(self, model: nn.Module):
        resp = []
        y = []
        for X, Y in self.data_loader:
            X = X.to(self.device)
            Y = Y.to(self.device)
            resp.append(model(X))
            y.append(Y)

        resp = torch.cat(resp)
        y = torch.cat(y)
        score = self.score_function(resp, y)

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
