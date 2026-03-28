import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import matplotlib.pyplot as plt
from datasets import load_dataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets from Hugging Face
datasets_mappings = {
    "astroML/linear_regression": "astroML",
    "cosmobg/cosmodataMG": "cosmobg",
    "holoviz/crash-causality": "holoviz",
}

# Use these datasets for synthetic cosmology examples
loaded_datasets = {
    name: load_dataset(k, split="train") for k, name in datasets_mappings.items()
}


# Define a custom dataset class for converting Hugging Face datasets to PyTorch
class CosmologyDataset(Dataset):
    def __init__(self, dataset, input_key, target_key):
        self.dataset = dataset
        self.input_key = input_key
        self.target_key = target_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        x = torch.FloatTensor(data[self.input_key]).unsqueeze(0)
        y = torch.FloatTensor(data[self.target_key]).unsqueeze(0)
        return x, y


# Prepare datasets
cosmology_datasets = {
    name: CosmologyDataset(ds, "feature", "value")
    for name, ds in loaded_datasets.items()
}


# Function to create model
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
    }
    for name in cosmology_datasets.keys()
}

# Hyperparameter tuning for hidden layer size
hidden_layer_size = 64
model = create_model(hidden_layer_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate on each dataset
for ds_name, dataset in cosmology_datasets.items():
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                experiment_data[ds_name]["predictions"].extend(outputs.cpu().numpy())
                experiment_data[ds_name]["ground_truth"].extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        experiment_data[ds_name]["losses"]["val"].append(val_loss)
        BER = np.exp(-val_loss)
        experiment_data[ds_name]["metrics"]["val"].append(BER)

        print(
            f"Epoch {epoch + 1}: Dataset = {ds_name}, Validation Loss = {val_loss:.4f}, BER = {BER:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
