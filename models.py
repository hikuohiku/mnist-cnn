import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Simple_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 畳み込み層1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # プーリング層1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 畳み込み層2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # プーリング層2
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # 全結合層1
            nn.ReLU(),
            nn.Linear(128, 10),  # 全結合層2（出力層）
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, 28, 28)  # 入力を28x28の画像形式に変換
        x = self.conv_layers(x)  # 畳み込み層を通過
        x = x.view(x.size(0), -1)  # 平坦化
        x = self.fc_layers(x)  # 全結合層を通過
        return x
