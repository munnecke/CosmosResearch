import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset loading
dataset_indices = [0, 1, 2]  # Choose three indices for the datasets
dataset_names = ["some_astro_dataset1", "some_astro_dataset2", "some_astro_dataset3"]
datasets = [load_dataset(name, split="train") for name in dataset_names]

# Preprocess and normalize datasets
scaler = MinMaxScaler(feature_range=(0, 1))

processed_data = []
for data in datasets:
    # Hypothetical preprocessing; adapt depending on dataset structure
    inputs = np.array([example["redshift"] for example in data])[:, np.newaxis]
    targets = np.array([example["measurement"] for example in data])[:, np.newaxis]

    # Normalize
    inputs_scaled = scaler.fit_transform(inputs)
    targets_scaled = scaler.fit_transform(targets)

    processed_data.append(
        (torch.FloatTensor(inputs_scaled), torch.FloatTensor(targets_scaled))
    )


# Define synthetic dataset structure
class CosmologyDataset(Dataset):
    def __init__(self, data):
        self.inputs, self.targets = data

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


# Setup data loaders
data_loaders = []
for data in processed_data:
    dataset = CosmologyDataset(data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    data_loaders.append((train_loader, val_loader))


# Model architecture
def create_model(hidden_layer_size):
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(1, hidden_layer_size),
                nn.ReLU(),
                nn.Linear(hidden_layer_size, hidden_layer_size),
                nn.ReLU(),
                nn.Linear(hidden_layer_size, 1),
            )

        def forward(self, x):
            return self.fc(x)

    return SimpleModel().to(device)


# Experiment data container
experiment_data = {
    name: {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "hidden_layer_sizes": [],
    }
    for name in dataset_names
}

# Hyperparameter tuning for each dataset
hidden_layer_sizes = [16, 32, 64, 128, 256]
for size in hidden_layer_sizes:
    for (train_loader, val_loader), dataset_name in zip(data_loaders, dataset_names):
        model = create_model(size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 20
        for epoch in range(num_epochs):
            model.train()
            training_loss = 0.0
            for batch in train_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                inputs, targets = batch

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                training_loss += loss.item()

            experiment_data[dataset_name]["losses"]["train"].append(
                training_loss / len(train_loader)
            )

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    inputs, targets = batch
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    # Save predictions and ground truth
                    experiment_data[dataset_name]["predictions"].extend(
                        outputs.cpu().numpy()
                    )
                    experiment_data[dataset_name]["ground_truth"].extend(
                        targets.cpu().numpy()
                    )

            val_loss /= len(val_loader)
            experiment_data[dataset_name]["losses"]["val"].append(val_loss)

            # Compute Bayesian Evidence Ratio (BER) as a placeholder value
            BER = np.exp(-val_loss)
            experiment_data[dataset_name]["metrics"]["val"].append(BER)

            print(
                f"Dataset: {dataset_name}, Epoch {epoch + 1}: Hidden Layer Size = {size}, Validation Loss = {val_loss:.4f}, BER = {BER:.4f}"
            )

        # Record the hidden_layer_size
        experiment_data[dataset_name]["hidden_layer_sizes"].append(size)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
