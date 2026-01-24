"""
Utility functions and classes for Jupyter Notebooks lessons. 
"""

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, random_split
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        return x


def train_model(model, train_set):
    batch_size = 64
    num_epochs = 10

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch in train_loader:
            inputs = batch["image"]
            labels = batch["label"]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}: Loss = {running_loss / len(train_loader)}")


    print("Training complete")
    return epoch_losses

def plot_loss_curve(losses, title):
    plt.figure(figsize=(4, 3))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.show()


def evaluate_model(model, test_set):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["image"]
            labels = batch["label"]

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()

    accuracy = correct / total
    average_loss = total_loss / len(test_loader)

    return average_loss, accuracy




def include_digits(dataset, included_digits):
    including_indices = [
        idx
        for idx in range(len(dataset))
        if dataset[idx]["label"] in included_digits
    ]
    return torch.utils.data.Subset(dataset, including_indices)


def exclude_digits(dataset, excluded_digits):
    including_indices = [
        idx
        for idx in range(len(dataset))
        if dataset[idx]["label"] not in excluded_digits
    ]
    return torch.utils.data.Subset(dataset, including_indices)


def plot_distribution(dataset, title):
    labels = [dataset[i]["label"] for i in range(len(dataset))]

    unique_labels, label_counts = torch.unique(
        torch.tensor(labels), return_counts=True
    )

    plt.figure(figsize=(4, 2))
    plt.bar(unique_labels.tolist(), label_counts.tolist())
    plt.title(title)
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.xticks(range(10))
    plt.show()



def compute_confusion_matrix(model, testset):
    true_labels = []
    predicted_labels = []

    model.eval()

    with torch.no_grad():
        for i in range(len(testset)):
            sample = testset[i]
            image = sample["image"].unsqueeze(0)  # add batch dim
            label = sample["label"]

            output = model(image)
            _, predicted = torch.max(output, 1)

            true_labels.append(label)
            predicted_labels.append(predicted.item())

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    cm = confusion_matrix(true_labels, predicted_labels)
    return cm



def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", linewidths=0.5)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def plot_confusion_matrix_with_missing(cm, title, missing_digits):
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="gray"
    )

    for d in missing_digits:
        ax.add_patch(
            plt.Rectangle((0, d), 10, 1, fill=False, edgecolor="red", linewidth=2)
        )

    plt.title(f"{title}\nMissing digits: {missing_digits}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()



def show_training_samples(dataset, title, samples_per_digit=3):
    # Collect samples per digit
    digit_samples = {d: [] for d in range(10)}

    for i in range(len(dataset)):
        sample = dataset[i]
        digit = sample["label"]

        if len(digit_samples[digit]) < samples_per_digit:
            digit_samples[digit].append(sample["image"])

    # Create figure
    fig, axes = plt.subplots(
        10, samples_per_digit, figsize=(samples_per_digit * 1.2, 6)
    )
    fig.suptitle(title, fontsize=12)

    for d in range(10):
        for j in range(samples_per_digit):
            ax = axes[d, j]
            if j < len(digit_samples[d]):
                ax.imshow(digit_samples[d][j].squeeze(), cmap="gray")
            ax.axis("off")

    plt.tight_layout()
    plt.show()