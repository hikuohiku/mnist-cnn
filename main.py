import torch
import torchvision
import torchvision.transforms as transforms
from dotenv import dotenv_values


class Config:
    def __init__(self, env):
        self.batch_size = int(env["batch_size"] or "64")
        self.test_batch_size = int(env["test_batch_size"] or "32")


def main(config: Config):
    # MNISTデータセットをダウンロード
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )
    print(train_dataset)
    print(test_dataset)
    print(train_dataset[0][0].shape)
    print(type(train_dataset[0][1]))

    # データローダーを作成
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=config.test_batch_size, shuffle=False
    )

    print("MNISTデータセットをロードしました。")


if __name__ == "__main__":
    env = dotenv_values(".env")
    config = Config(env)
    main(config)
