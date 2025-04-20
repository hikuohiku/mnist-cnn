import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def run_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
):
    model.train()  # モデルを学習モードに設定
    for data, target in tqdm(train_loader):
        optimizer.zero_grad()  # 勾配を初期化
        output = model(data)  # モデルにデータを入力
        # print(output.shape)  # torch.Size([64, 10])
        loss = criterion(output, target)  # 損失を計算
        # print(loss)  # tensor(2.1018, grad_fn=<NllLossBackward0>)
        loss.backward()  # 勾配を計算
        optimizer.step()  # パラメータを更新
