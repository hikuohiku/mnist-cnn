import datetime

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
        self.experiment_id = env.get(
            "experiment_id", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.train_id = 0
        self.train_count = int(env["train_count"] or "1")

    def next_training(self):
        self.train_id += 1

    def to_dict(self):
        return {
            "batch_size": self.batch_size,
            "test_batch_size": self.test_batch_size,
            "lr": self.lr,
            "epochs": self.epochs,
            "use_cuda": self.use_cuda,
            "device": str(self.device),
            "experiment_id": self.experiment_id,
            "train_id": self.train_id,
        }
