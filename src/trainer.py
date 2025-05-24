import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import datetime
from pathlib import Path
import logging

# Установка логгирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def MNIST(batch_size=64, sample_size=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if sample_size is not None:
        indices = torch.randperm(len(train_dataset)).tolist()[:sample_size]
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader


class Trainer:
    def __init__(self, dataset_name, batch_size, model, checkpoint_path, device=None):
        assert dataset_name == 'MNIST', "Only MNIST supported currently"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.checkpoint_path = Path(checkpoint_path)

        self.train_dataset, self.test_dataset, self.train_loader, self.test_loader = MNIST(batch_size)

        self.history = {"train_loss": [], "test_loss": [], "test_accuracy": []}
        self.hyperparams = {
            "model_class": model.__class__.__name__,
            "input_dim": model.model[0].in_features,
            "output_dim": model.model[-1].out_features,
            "num_params": sum(p.numel() for p in model.parameters()),
            "timestamp": datetime.datetime.now().isoformat()
        }

        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        return test_loss, accuracy

    def train(self, n_epochs, lr=1e-3):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(self.train_loader)
            test_loss, test_accuracy = self.evaluate()

            self.history["train_loss"].append(avg_train_loss)
            self.history["test_loss"].append(test_loss)
            self.history["test_accuracy"].append(test_accuracy)

            logging.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}, Accuracy = {test_accuracy:.2f}%")

        # Сохраняем модель на CPU (чтобы избежать проблем с загрузкой на другом устройстве)
        model_path = self.checkpoint_path / "model.pt"
        torch.save(self.model.to('cpu').state_dict(), model_path)
        self.model.to(self.device)  # возвращаем на устройство для дальнейшей работы

        with open(self.checkpoint_path / "hyperparams.json", "w") as f:
            json.dump(self.hyperparams, f, indent=4)

        with open(self.checkpoint_path / "history.json", "w") as f:
            json.dump(self.history, f, indent=4)

        logging.info(f"Model and logs saved to {self.checkpoint_path}")

    def evaluate_model(self, model_to_eval, description="Model"):
        model_to_eval.to(self.device)
        model_to_eval.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model_to_eval(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        print(f"{description}: Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.2f}%")
        return test_loss, accuracy
