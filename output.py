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
    plt.ylim(0, 1.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.ylim(75, 100)
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


def calculate_mean_and_variance(csv_path):
    """
    CSVファイルからtrain_loss, train_accuracy, test_loss, test_accuracyの平均と分散を計算する。
    """
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            train_losses.append(float(row["train_loss"]))
            train_accuracies.append(float(row["train_accuracy"]))
            test_losses.append(float(row["test_loss"]))
            test_accuracies.append(float(row["test_accuracy"]))

    train_losses_mean = np.mean(train_losses)
    train_losses_variance = np.var(train_losses)
    train_accuracies_mean = np.mean(train_accuracies)
    train_accuracies_variance = np.var(train_accuracies)
    test_losses_mean = np.mean(test_losses)
    test_losses_variance = np.var(test_losses)
    test_accuracies_mean = np.mean(test_accuracies)
    test_accuracies_variance = np.var(test_accuracies)

    return (
        train_losses_mean,
        train_losses_variance,
        train_accuracies_mean,
        train_accuracies_variance,
        test_losses_mean,
        test_losses_variance,
        test_accuracies_mean,
        test_accuracies_variance,
    )


def save_summary(config: Config):
    save_dir = os.path.join("results", config.experiment_id)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "result.csv")
    (
        train_losses_mean,
        train_losses_variance,
        train_accuracies_mean,
        train_accuracies_variance,
        test_losses_mean,
        test_losses_variance,
        test_accuracies_mean,
        test_accuracies_variance,
    ) = calculate_mean_and_variance(csv_path)

    print(f"Train Loss Mean: {train_losses_mean}")
    print(f"Train Loss Variance: {train_losses_variance}")
    print(f"Train Accuracy Mean: {train_accuracies_mean}")
    print(f"Train Accuracy Variance: {train_accuracies_variance}")
    print(f"Test Loss Mean: {test_losses_mean}")
    print(f"Test Loss Variance: {test_losses_variance}")
    print(f"Test Accuracy Mean: {test_accuracies_mean}")
    print(f"Test Accuracy Variance: {test_accuracies_variance}")

    # 結果をファイルに保存
    result_path = os.path.join("results", config.experiment_id, "summary.txt")
    with open(result_path, "w") as f:
        f.write(f"Train Loss Mean: {train_losses_mean}\n")
        f.write(f"Train Loss Variance: {train_losses_variance}\n")
        f.write(f"Train Accuracy Mean: {train_accuracies_mean}\n")
        f.write(f"Train Accuracy Variance: {train_accuracies_variance}\n")
        f.write(f"Test Loss Mean: {test_losses_mean}\n")
        f.write(f"Test Loss Variance: {test_losses_variance}\n")
        f.write(f"Test Accuracy Mean: {test_accuracies_mean}\n")
        f.write(f"Test Accuracy Variance: {test_accuracies_variance}\n")
