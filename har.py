import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import random
import os
import argparse
from pathlib import Path

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

class HARModel(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(HARModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout_rate * 0.8)

        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 6)

    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(torch.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(torch.relu(self.bn4(self.fc4(x))))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def load_data(train_path, test_path, n_samples=None):
    train_data = pd.read_csv(train_path)

    if n_samples:
        train_data = train_data.head(n_samples)

    X_train = train_data.drop('y', axis=1)
    y_train = train_data.y - 1

    X_test = pd.read_csv(test_path, index_col='id')

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, y_train, X_test

def train_model(model, train_loader, val_loader, epochs, device, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        history['val_acc'].append(accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}%")

    return history

def predict(model, X_test, device):
    model.eval()

    X_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(X_tensor)
        probabilities = torch.softmax(predictions, dim=1)
        predicted_labels = torch.max(probabilities, 1)[1]

    return predicted_labels.cpu().numpy() + 1, probabilities.cpu().numpy()

def save_predictions(predictions, output_path):
    result = pd.DataFrame(predictions, columns=['y'])
    result.to_csv(output_path, index_label='id')
    print(f"Predictions saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train HAR model and make predictions')
    parser.add_argument('--train', type=str, default='trainYX.csv')
    parser.add_argument('--test', type=str, default='testX.csv')
    parser.add_argument('--output', type=str, default='HAR_baseline.csv')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--val_split', type=float, default=0.3)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=0.3)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_train, y_train, X_test = load_data(args.train, args.test, args.n_samples)

    X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = HARModel(input_size=X_train.shape[1], dropout_rate=args.dropout).to(device)
    print(f"Model input size: {X_train.shape[1]}")

    history = train_model(model, train_loader, val_loader, args.epochs, device, args.lr)
    predictions, probabilities = predict(model, X_test, device)
    save_predictions(predictions, args.output)

if __name__ == "__main__":
    main()