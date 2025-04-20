import torch
from torch import cuda


class Config:
    def __init__(self, env):
        self.batch_size = int(env["batch_size"] or "64")
        self.test_batch_size = int(env["test_batch_size"] or "32")
        self.lr = float(env["lr"] or "0.1")
        self.epochs = int(env["epochs"] or "10")
        self.use_cuda = env["use_cuda"] == "True"
        use_cuda = self.use_cuda and cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
