import torch
import torch.nn as nn


def base_extractor(in_dim, hidden_dim, activation_fn=nn.Sigmoid):
    return nn.Sequential(nn.Linear(in_dim, hidden_dim),
                         activation_fn())


class ELM(nn.Module):

    def __init__(self, extractor: nn.Module, nb_classes: int):
        super(ELM, self).__init__()
        self.H = extractor
        self.out_layer = None
        self.is_initialised = False
        self.nb_classes = nb_classes

    def initialise_out_layer(self, hidden_dim, bias=False, device='cpu', softmax=False):
        self.out_layer = torch.nn.Linear(hidden_dim,
                                         self.nb_classes,
                                         bias=bias).to(device)
        if softmax:
            self.out_layer = nn.Sequential(self.out_layer, nn.Softmax())
        self.is_initialised = True

    def initialise_xavier_normal(self, hidden_dim):
        self.H.apply(lambda l: nn.init.xavier_normal(l.weight))
        self.initialise_out_layer(hidden_dim)
        nn.init.xavier_normal(self.out_layer)

    def forward(self, x):
        H = self.H(x)
        H = H.reshape(H.shape[0], -1)
        if not self.is_initialised:
            self.initialise_out_layer(hidden_dim=H.shape[1],
                                      device='cuda' if H.is_cuda else 'cpu')
        return self.out_layer(H)

    def predict(self, x):
        return torch.softmax(self.forward(x), 1)

    def set_up_out_layer(self, out_layer: nn.Linear):
        self.out_layer = out_layer
        self.is_initialised = True


class OutputLayerELM(nn.Module):

    def __init__(self, nb_classes):
        super(OutputLayerELM, self).__init__()
        self.is_initialised = False
        self.dense = None
        self.nb_classes = nb_classes

    def forward(self, H):
        H = H.reshape(H.shape[0], -1)

        if not self.is_initialised:
            self.dense = torch.nn.Linear(H.shape[1], self.nb_classes, bias=False)
            self.is_initialised = True
        return self.dense(H)
