"""
task.py
Shared ML utilities for MNIST federated learning.
"""


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

import numpy as np

import seaborn as sns
from sklearn.metrics import confusion_matrix


# -------------------------------------------------
# Model
# -------------------------------------------------
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)
        return self.out(x)


# -------------------------------------------------
# Data transforms
# -------------------------------------------------
_transform = transforms.ToTensor()

def apply_transforms(batch):
    batch["image"] = [_transform(img) for img in batch["image"]]
    return batch


# -------------------------------------------------
# Dataset helpers
# -------------------------------------------------
def exclude_digits(dataset, excluded_digits):
    indices = [
        i for i in range(len(dataset))
        if dataset[i]["label"] not in excluded_digits
    ]
    return Subset(dataset, indices)


# -------------------------------------------------
# Client datasets (3 clients)
# -------------------------------------------------
def load_client_datasets():
    """
    Load MNIST once and split into three client datasets,
    each missing a different set of digits.
    """

    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"train": IidPartitioner(num_partitions=1)},
    )

    trainset = fds.load_partition(0).with_transform(apply_transforms)

    total_len = len(trainset)
    split_len = total_len // 3

    torch.manual_seed(42)
    part1, part2, part3 = random_split(trainset, [split_len] * 3)

    part1 = exclude_digits(part1, [1, 3, 7])
    part2 = exclude_digits(part2, [2, 5, 8])
    part3 = exclude_digits(part3, [4, 6, 9])

    return part1, part2, part3


# -------------------------------------------------
# Test datasets
# -------------------------------------------------
def load_test_sets():
    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"test": IidPartitioner(num_partitions=1)},
    )

    testset = fds.load_partition(0).with_transform(apply_transforms)

    test_137 = exclude_digits(testset, [0, 2, 4, 5, 6, 8, 9])
    test_258 = exclude_digits(testset, [0, 1, 3, 4, 6, 7, 9])
    test_469 = exclude_digits(testset, [0, 1, 2, 3, 5, 7, 8])

    return testset, test_137, test_258, test_469


# -------------------------------------------------
# Training
# -------------------------------------------------
def train(model, trainloader, epochs, device):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    running_loss = 0.0

    for _ in range(epochs):
        for batch in trainloader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    return running_loss / len(trainloader)


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def test(model, dataloader, device):
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            outputs = model(x)
            loss += criterion(outputs, y).item()

            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return loss / len(dataloader), correct / total

def evaluate_model(model, dataset, device):
    """
    Server-side evaluation helper.
    Wraps `test` so the server can call a simple API.
    """
    dataloader = DataLoader(dataset, batch_size=64)
    _, accuracy = test(model, dataloader, device)
    return accuracy


def compute_confusion_matrix(model, testset, device):
    model.eval()

    y_true = []
    y_pred = []

    loader = DataLoader(testset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(
    cm,
    title,
    filename="confusion_matrix_missing.png",
):
    # Normalize per true label and convert to percentage
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_pct = np.nan_to_num(cm_pct) * 100

    plt.figure(figsize=(8, 6))

    ax = sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Greys",
        cbar=True,
        linewidths=0.5,
        linecolor="lightgray",
        vmin=0,
        vmax=100,
        annot_kws={"size": 11},
    )

    

    ax.set_title(
        f"{title}\nRows = true digit, columns = predicted digit (values in %)",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")