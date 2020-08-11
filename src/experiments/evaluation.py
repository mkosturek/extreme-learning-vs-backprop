from sklearn.metrics import (accuracy_score,
                             f1_score,
                             recall_score,
                             precision_score)
from torch.utils.data import DataLoader
from sklearn.metrics.regression import mean_squared_error, mean_absolute_error
import numpy as np
import torch


class Evaluator:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    @torch.no_grad()
    def evaluate(self, model):
        raise NotImplementedError


class ClassificationEvaluator(Evaluator):

    def __init__(self, X, y, average=None, nb_classes=None):
        super().__init__(X, y)

        if average is not None:
            self.average = average
        elif nb_classes is not None:
            self.average = 'binary' if nb_classes == 2 else 'weighted'
        elif len(set(y.numpy().ravel())) > 2:
            self.average = 'weighted'
        else:
            self.average = 'binary'

    @torch.no_grad()
    def evaluate(self, model):
        response = (model.predict(self.X).argmax(1)
                    .cpu().detach().numpy().ravel())
        return evaluate_classification(self.y.cpu().numpy().ravel(),
                                       response,
                                       self.average)


class RegressionEvaluator(Evaluator):

    def __init__(self, X, y):
        super().__init__(X, y)

    @torch.no_grad()
    def evaluate(self, model):
        response = model(self.X).cpu().detach().numpy().ravel()
        return evaluate_regression(self.y.cpu().detach().numpy().ravel(), response)


class DataLoaderEvaluator:

    def __init__(self, data_loader: DataLoader, device='cpu'):
        self.data_loader = data_loader
        self.device = device

    @torch.no_grad()
    def evaluate(self, model):
        raise NotImplementedError


class ClassificationDataLoaderEvaluator(DataLoaderEvaluator):

    def __init__(self, data_loader, average=None, nb_classes=None, device='cpu'):
        super().__init__(data_loader, device)
        self.nb_classes = nb_classes
        self.average = average
    
    @torch.no_grad()
    def evaluate(self, model):
        response = []
        y = []
        for X, target in self.data_loader:
            response.append(model.predict(X.to(self.device)).argmax(1)
                            .cpu().detach().numpy().ravel())
            y.append(target.cpu().detach().numpy().ravel())

        response = np.concatenate(response)
        y = np.concatenate(y)
        
        if self.average is None:
            if self.nb_classes is not None:
                self.average = ('binary' 
                                if self.nb_classes == 2 
                                else 'weighted')
            elif len(set(y)) > 2:
                self.average = 'weighted'
            else:
                self.average = 'binary'

        return evaluate_classification(y, response, self.average)


class RegressionDataLoaderEvaluator(DataLoaderEvaluator):

    def __init__(self, data_loader, device='cpu'):
        super().__init__(data_loader, device)

    @torch.no_grad()
    def evaluate(self, model):

        response = []
        y = []
        for X, target in self.data_loader:
            response.append(model(X.to(self.device)).cpu()
                                  .detach().numpy().ravel())
            y.append(target.cpu().detach().numpy().ravel())

        response = np.concatenate(response)
        y = np.concatenate(y)
        # response = model(self.X).cpu().detach().numpy().ravel()
        return evaluate_regression(y, response)



def evaluate_classification(y_true, y_pred, average=None):
    if average is None:
        if max(len(set(y_true)), len(set(y_pred))) > 2:
            average = 'weighted'
        else:
            average = 'binary'

    return {'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average=average),
            'recall': recall_score(y_true, y_pred, average=average),
            'precision': recall_score(y_true, y_pred, average=average)}


def evaluate_regression(y_true, y_pred):
    return {'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred)}
