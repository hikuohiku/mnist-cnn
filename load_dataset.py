import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


def load_mnist(config) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [0, 255] → [0.0, 1.0]
            transforms.Lambda(lambda x: x.view(-1)),  # 28x28 → 784ベクトルへ変換
        ]
    )
    # MNISTデータセットを取得
    train_dataset: Dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset: Dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    # データローダーを作成
    train_loader: DataLoader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_loader: DataLoader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=config.test_batch_size, shuffle=False
    )

    return train_loader, test_loader
