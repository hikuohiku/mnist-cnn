import torch
import torchvision
import torchvision.transforms as transforms


def main():
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
        dataset=train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=64, shuffle=False
    )

    print("MNISTデータセットをロードしました。")


if __name__ == "__main__":
    main()
