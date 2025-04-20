import time

from dotenv import dotenv_values
from torch import nn, optim

from config import Config
from eval import evaluate
from load_dataset import load_mnist
from models import Simple_CNN
from output import plot, save_config, save_results, save_summary
from train import run_epoch


def main(config: Config):
    # MNISTデータセットをロード
    train_loader, test_loader = load_mnist(config)

    # モデルの定義
    model = Simple_CNN()
    model.to(config.device)
    print(model) if config.train_id == 0 else None

    # 損失関数と最適化手法の定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    # 学習の実行
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    start_time = time.time()

    for epoch in range(config.epochs):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, config
        )
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, config)
        print(
            "Epoch {}: Train set: Average loss: {:.4f}, Accuracy: {:.2f}%".format(
                epoch + 1, train_loss, train_accuracy
            )
        )
        print(
            "Epoch {}: Test set: Average loss: {:.4f}, Accuracy: {:.2f}%".format(
                epoch + 1, test_loss, test_accuracy
            )
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    end_time = time.time()
    training_time = end_time - start_time
    print("Training time: {:.2f} seconds".format(training_time))

    # 学習結果のプロット
    plot(train_losses, test_losses, train_accuracies, test_accuracies, config)  # type: ignore
    # configの保存
    save_config(config)
    # 学習結果の保存
    save_results(
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies,
        training_time,
        config,
    )


if __name__ == "__main__":
    env = dotenv_values(".env")
    config: Config = Config(env)
    print("Using device:", config.device)
    for i in range(config.train_count):
        print(f"\n[Training {i + 1}]\n")
        main(config)
        config.next_training()
    # 学習サマリーの保存
    save_summary(config)
