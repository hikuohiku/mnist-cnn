import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from config import Config


def plot(train_losses, test_losses, train_accuracies, test_accuracies, config: Config):
    epochs = np.arange(1, config.epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()

    plt.tight_layout()
    save_dir = os.path.join("results", config.experiment_id)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"loss_accuracy_{config.train_id}.svg")
    plt.savefig(path)
    plt.show()


def save_results(
    train_losses,
    test_losses,
    train_accuracies,
    test_accuracies,
    training_time,
    config: Config,
):
    # CSVファイルのパスを生成
    save_dir = os.path.join("results", config.experiment_id)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "result.csv")

    # ヘッダー
    header = [
        "train_id",
        "training_time",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
    ]

    # ファイルが存在しない場合はヘッダーを書き込む
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)

        # 最終的な損失と精度
        train_loss = train_losses[-1]
        test_loss = test_losses[-1]
        train_accuracy = train_accuracies[-1]
        test_accuracy = test_accuracies[-1]

        # データをCSVファイルに追記
        writer.writerow(
            [
                config.train_id,
                training_time,
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy,
            ]
        )


def save_config(config: Config):
    import json

    # JSONファイルのパスを生成
    save_dir = os.path.join("results", config.experiment_id)
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")

    # configをJSONファイルに保存
    with open(config_path, "w") as jsonfile:
        json.dump(config.to_dict(), jsonfile, indent=4)
