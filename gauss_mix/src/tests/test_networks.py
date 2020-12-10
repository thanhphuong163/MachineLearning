import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vae.networks import Dense, NeuralNetwork, GaussianNet, GumbelSoftmax 

class TestNetworks(unittest.TestCase):
    def setUp(self):
        self.in_ = 2
        self.out_ = 3
        self.x = torch.randn((3, 2))
        return super().setUp()

    def test_dense_wo_activation(self):
        dense = Dense(self.in_, self.out_, bias=False, activation=None)
        weights = dense.parameters()
        x_ = None
        for weight in weights:
            x_ = self.x @ weight.T
        assert_result = torch.equal(dense(self.x), x_)
        self.assertEqual(assert_result, True, 'Should be equal.')

    def test_dense_w_activation(self):
        dense = Dense(self.in_, self.out_, bias=False, activation='relu')
        params = dense.parameters()
        x_ = self.x
        for param in params:
            # print(f"\n{param})
            x_ = x_ @ param.T
        x_ = nn.functional.relu(x_)
        assert_result = torch.equal(dense(self.x), x_)
        # print(x_)
        self.assertEqual(assert_result, True, 'Should be equal.')

    def test_nn_sigmoid(self):
        net = NeuralNetwork(layers_specs=[2,3,1], layers_activations=['sigmoid','sigmoid'], bias=False)
        params = net.parameters()
        y = self.x
        for i, param in enumerate(params):
            # print(f"\n{param})
            y = y @ param.T
            y = torch.sigmoid(y)
        # print(f"\n{y})
        result = torch.equal(net(self.x), y)
        self.assertEqual(result, True, "Should be equal.")

    def test_gaussian_net(self):
        net = {
            'layers_specs': [2,3,2],
            'layers_activations': ['relu', None],
            'bias': False
        }
        gauss_net = GaussianNet(common_net=net, mean_net=net, var_net=net)
        params = gauss_net.parameters()
        # for param in params:
            # print(f"\n{param}")
        mean, var, z = gauss_net(self.x)
        # print(f"Mean:\n{mean}")
        # print(f"Variance:\n{var}")
        # print(f"z:\n{z}")
        self.assertEqual(True, True, "Should be equal")

    def test_gumbel_softmax(self):
        net = {
            'layers_specs': [2,3,2],
            'layers_activations': ['relu', None],
            'bias': False
        }
        gumbel_net = GumbelSoftmax(net=net)
        Y = gumbel_net(self.x)
        print(f"Y:\n{Y}")
        self.assertEqual(True, True)

    def test_gumbel_softmax_hard(self):
        net = {
            'layers_specs': [2,3,2],
            'layers_activations': ['relu', None],
            'bias': False
        }
        gumbel_net = GumbelSoftmax(net=net, hard=True)
        Y = gumbel_net(self.x)
        print(f"Y:\n{Y}")
        self.assertEqual(True, True)

    def test_infer_net(self):
        net = {
            'layers_specs': [2,3,2],
            'layers_activations': ['relu', None],
            'bias': False
        }

















        
