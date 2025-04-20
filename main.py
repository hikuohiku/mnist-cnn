from dotenv import dotenv_values
from torch import nn, optim

from eval import evaluate
from load_dataset import load_mnist
from models import MLP
from train import run_epoch


class Config:
    def __init__(self, env):
        self.batch_size = int(env["batch_size"] or "64")
        self.test_batch_size = int(env["test_batch_size"] or "32")
        self.lr = float(env["lr"] or "0.1")
        self.epochs = int(env["epochs"] or "10")


def main(config: Config):
    # MNISTデータセットをロード
    train_loader, test_loader = load_mnist(config)

    # モデルの定義
    model = MLP()
    print(model)

    # 損失関数と最適化手法の定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    # 学習の実行
    for _ in range(config.epochs):
        run_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {:.2f}%".format(
                test_loss, test_accuracy
            )
        )


if __name__ == "__main__":
    env = dotenv_values(".env")
    config: Config = Config(env)
    main(config)
