import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):
    def __init__(self, in_, out_, bias=True, activation=None):
        super(Dense, self).__init__()
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(),
            'soflplus': nn.Softplus()
        }
        fc = nn.Linear(in_, out_, bias=bias)
        activation = None if activation == None else activations[activation]
        self.model = nn.Sequential(fc, activation) if activation else nn.Sequential(fc)
    
    def forward(self, X):
        y = self.model(X)
        return y


class NeuralNetwork(nn.Module):
    def __init__(self,
                layers_specs=[2,3,1],
                layers_activations=['relu','relu'],
                bias=True):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        in_out_act = zip(layers_specs[:-1],
                        layers_specs[1:],
                        layers_activations)
        for in_, out_, act_ in in_out_act:
            layers.append(Dense(in_, out_, bias=bias, activation=act_))
        
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        y = self.model(X)
        return y


# Default networks for GaussianNet
COMMON_NET_DEFAULT = {
    "layers_specs": [2,3,1],
    "layers_activations": ['relu', 'relu'],
    "bias": False
}
MEAN_NET_DEFAULT = {
    "layers_specs": [2,3,1],
    "layers_activations": ['relu', None],
    "bias": False
}
VAR_NET_DEFAULT = {
    "layers_specs": [2,3,1],
    "layers_activations": ['relu', None],
    "bias": False
}


class GaussianNet(nn.Module):
    def __init__(self,
                common_net=COMMON_NET_DEFAULT,
                mean_net=MEAN_NET_DEFAULT,
                var_net=VAR_NET_DEFAULT):
        super(GaussianNet, self).__init__()
        self.common_net = NeuralNetwork(**common_net)
        self.mean_net = NeuralNetwork(**mean_net)
        self.var_net = NeuralNetwork(**var_net)

    def reparameterize(self, mu, var):
        assert torch.all(var >= 0), "Variance must be non-negative"
        std = torch.sqrt(var + 1e-10)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z
    
    def forward(self, X):
        y = self.common_net(X)
        mean = self.mean_net(y)
        var = F.softplus(self.var_net(y))
        z = self.reparameterize(mean, var)
        return mean, var, z


class GumbelSoftmax(nn.Module):
    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        # TODO: define GumbelSoftmax
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, shape, eps=1e-20):
        u = torch.rand(shape)
        return -torch.log(-torch.log(u + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.shape)
        return F.softmax(y/temperature)

    def forward(self, X):
        # TODO: define dataflow in GumbelSoftmax
        logits = self.logits(X)
        y = self.gumbel_softmax_sample(logits, temperature)

        return y


class StickBreakingNet(nn.Module):
    def __init__(self):
        super(StickBreakingNet, self).__init__()
        # TODO: define StickBreakingNet

    def forward(self, X):
        # TODO: define dataflow in StickBreakingNet
        return 0


# default networks for InferenceNet
GAUSSIAN_NET_DEFAULT = {
    'common_net': COMMON_NET_DEFAULT,
    'mean_net': MEAN_NET_DEFAULT,
    'var_net': VAR_NET_DEFAULT
}
GUMBEL_NET_DEFAULT = {}
STICK_BREAKING_NET_DEFAULT = {}


class InferenceNet(nn.Module):
    def __init__(self, 
                gaussian_net=GAUSSIAN_NET_DEFAULT,
                gumbel_net=GUMBEL_NET_DEFAULT,
                stick_breaking_net=STICK_BREAKING_NET_DEFAULT):
        super(InferenceNet, self).__init__()
        # TODO : declare inference networks here, including gauss_net, gumbel_net, stick_breaking_net
        
    def forward(self, X):
        # TODO: define dataflow inside InferenceNet
        return 0


class GenerativeNet(nn.Module):
    def __init__(self):
        super(GenerativeNet, self).__init__()
        # TODO : declare generative networks here, including gauss_net, gumbel_net, stick_breaking_net

    def forward(self, X):
        # TODO: define dataflow inside GenerativeNet
        return 0
