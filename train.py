import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config


def run_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    config: Config,
):
    model.train()  # モデルを学習モードに設定
    total_loss = 0
    correct = 0
    for data, target in tqdm(train_loader):
        data, target = (
            data.to(config.device),
            target.to(config.device),
        )  # データとターゲットをデバイスに転送
        optimizer.zero_grad()  # 勾配を初期化
        output = model(data)  # モデルにデータを入力
        loss = criterion(output, target)  # 損失を計算
        loss.backward()  # 勾配を計算
        optimizer.step()  # パラメータを更新

        total_loss += loss.item()  # バッチの損失を累積
        pred = output.argmax(dim=1, keepdim=True)  # スコアが最大値のインデックスを取得
        correct += pred.eq(target.view_as(pred)).sum().item()  # 正解数をカウント

    avg_loss = total_loss / len(train_loader)  # バッチの数でわる
    accuracy = 100 * correct / len(train_loader.dataset)  # type: ignore  # サンプル数でわる

    return avg_loss, accuracy
