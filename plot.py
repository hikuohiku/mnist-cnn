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
