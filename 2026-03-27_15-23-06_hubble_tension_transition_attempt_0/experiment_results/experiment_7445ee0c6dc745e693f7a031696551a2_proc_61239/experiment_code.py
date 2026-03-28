import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define synthetic dataset
class CosmologyDataset(Dataset):
    def __init__(self, size=1000):
        self.data = np.random.uniform(0, 2, (size, 1))  # Simulated redshifts
        # Simulated cosmological values, e.g., H0(z), w(z)
        self.targets = np.sin(self.data) + 0.1 * np.random.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), torch.FloatTensor(
            self.targets[index]
        )


# Create dataset and split into train/val
dataset = CosmologyDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)


# Experiment setup
optimizers_list = ["Adam", "RMSprop", "SGD"]
experiment_data = {
    f"optimizer_{opt}": {
        "synthetic_cosmology": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    for opt in optimizers_list
}

# Training loop for hyperparameters
for optimizer_name in optimizers_list:
    # Initialize model, criterion, optimizer
    model = SimpleModel().to(device)
    criterion = nn.MSELoss()

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):  # Fixed number of epochs for each optimizer
        model.train()
        training_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        experiment_data[f"optimizer_{optimizer_name}"]["synthetic_cosmology"]["losses"][
            "train"
        ].append(training_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Save predictions and ground truth
                experiment_data[f"optimizer_{optimizer_name}"]["synthetic_cosmology"][
                    "predictions"
                ].extend(outputs.cpu().numpy())
                experiment_data[f"optimizer_{optimizer_name}"]["synthetic_cosmology"][
                    "ground_truth"
                ].extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        experiment_data[f"optimizer_{optimizer_name}"]["synthetic_cosmology"]["losses"][
            "val"
        ].append(val_loss)

        # Compute Bayesian Evidence Ratio (BER) as a placeholder value (actual calculation would be more complex)
        BER = np.exp(
            -val_loss
        )  # Simplified as an illustrative placeholder, this isn't a proper BER calculation
        experiment_data[f"optimizer_{optimizer_name}"]["synthetic_cosmology"][
            "metrics"
        ]["val"].append(BER)

        print(
            f"Optimizer {optimizer_name}, Epoch {epoch + 1}: validation_loss = {val_loss:.4f}, BER = {BER:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
