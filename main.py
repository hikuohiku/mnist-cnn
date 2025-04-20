from dotenv import dotenv_values

from load_dataset import load_mnist
from models import MLP


class Config:
    def __init__(self, env):
        self.batch_size = int(env["batch_size"] or "64")
        self.test_batch_size = int(env["test_batch_size"] or "32")


def main(config: Config):
    # MNISTデータセットをロード
    train_loader, test_loader = load_mnist(config)

    # モデルの定義
    model = MLP()
    print(model)


if __name__ == "__main__":
    env = dotenv_values(".env")
    config = Config(env)
    main(config)
