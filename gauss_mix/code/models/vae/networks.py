import torch
import torch.nn as nn


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
        # TODO : define layers for neural network
        in_out_act = zip(layers_specs[:-1],
                        layers_specs[1:],
                        layers_activations)
        for in_, out_, act_ in in_out_act:
            layers.append(Dense(in_, out_, bias=bias, activation=act_))
        
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        # TODO : tell network how data can be forwarded
        y = self.model(X)
        return y

class InferenceNet(nn.Module):
    def __init__(self, 
                common_net: dict,
                mean_net: dict,
                log_var_net: dict):
        super(InferenceNet, self).__init__()
        # TODO : declare inference networks here, including gauss_net, gumbel_net, stick_breaking_net

class GenerativeNet(nn.Module):
    def __init__(self):
        super(GenerativeNet, self).__init__()
        # TODO : declare generative networks here, including gauss_net, gumbel_net, stick_breaking_net

