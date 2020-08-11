import torch
from src.elm.base_elm import ELM
from torch.utils.data import DataLoader


class ExtremeLearningAlgorithm:

    def __init__(self, C, use_gpu=True):
        self.C = C
        self.use_gpu = use_gpu



    @torch.no_grad()
    def train_data_loader(self, model: ELM, data_loader: DataLoader, 
                          W=None, try_pinverse=False):
        H = []
        Y = []
        if self.use_gpu:
            model = model.to('cuda')
        for x, y in data_loader:
            if self.use_gpu:
                y = y.to('cuda')
                x = x.to('cuda')
            if len(y.shape) == 1:
                y = y.view(-1,1)
            if y.shape[1] != model.nb_classes:
                # print(y)
                y = self.convert_y_to_onehot(model, y).to('cpu')
            H.append(model.H(x).to('cpu'))
            Y.append(y.to('cpu'))
        
        H = torch.cat(H)
        H = H.reshape(H.shape[0], -1)
        Y = torch.cat(Y)
        model = model.to('cpu')
        self._train_with_encoding(model, H, Y, W, try_pinverse)

    @torch.no_grad()
    def train(self, model: ELM, x, y, W=None, try_pinverse=False):
        H = model.H(x)

        if len(y.shape) == 1:
            y = y.view(-1,1)
        if y.shape[1] != model.nb_classes:
            y = self.convert_y_to_onehot(model, y)

        H = H.reshape(H.shape[0], -1)

        self._train_with_encoding(model, H, y, W, try_pinverse)

    @torch.no_grad()
    def _train_with_encoding(self, model: ELM, H: torch.FloatTensor, y, 
                             W=None, try_pinverse=False):

        if W is None:
            W = torch.ones((len(y), 1))
        
        HT = H.t()
        if H.shape[1] <= H.shape[0]:
            regul = self.C * torch.eye(H.shape[1])
            try:
                W_out = torch.inverse(regul + HT @ (W * H)) @ HT @ (W * y)
            except RuntimeError as e:
                if try_pinverse:
                    W_out = torch.pinverse(regul + HT @ (W * H)) @ HT @ (W * y)
                else:
                    raise e
        else:
            regul = self.C * torch.eye(H.shape[0])
            try:
                W_out = HT @ torch.inverse(regul + ((W * H) @ HT)) @ (W * y)
            except RuntimeError as e:
                if try_pinverse:
                    W_out = HT @ torch.pinverse(regul + ((W * H) @ HT)) @ (W * y)
                else:
                    raise e
        dense = torch.nn.Linear(H.shape[1], model.nb_classes, bias=False)
        dense.weight.data = torch.nn.Parameter(W_out.t())
        model.set_up_out_layer(dense)

    @staticmethod
    def convert_y_to_onehot(model, y):
        y_onehot = torch.FloatTensor(y.shape[0], model.nb_classes).to('cuda' if next(model.parameters()).is_cuda else 'cpu')
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)
        y = y_onehot
        return y

