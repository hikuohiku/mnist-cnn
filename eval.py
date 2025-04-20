import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module, test_loader: DataLoader, criterion: nn.Module
) -> tuple[float, float]:
    model.eval()  # モデルを評価モードに設定
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 勾配計算を無効にする
        for data, target in test_loader:
            output = model(data)  # モデルにデータを入力
            test_loss += criterion(output, target).item()  # 損失を計算
            pred = output.argmax(
                dim=1, keepdim=True
            )  # スコアが最大値のインデックスを取得
            correct += pred.eq(target.view_as(pred)).sum().item()  # 正解数をカウント

    test_loss /= len(test_loader)  # バッチの数でわる
    accuracy = 100.0 * correct / len(test_loader.dataset)  # type: ignore  # サンプル数でわる
    return test_loss, accuracy
