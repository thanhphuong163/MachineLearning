# TODO: Define variational autoencoder for Gaussian Mixture model
import torch


class GMVAE(object):
    def __init__(self, args):
        super(GMVAE, self).__init__()
        self.batch_size = args.batch_size
        self.gpu = args.gpu
        self.epoch = args.epoch

    def dataloader(self, X):


    def __call__(self, X):
        return 0
