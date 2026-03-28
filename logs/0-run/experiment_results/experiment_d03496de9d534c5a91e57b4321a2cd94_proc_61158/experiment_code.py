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
        self.targets = np.sin(self.data) + 0.1 * np.random.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), torch.FloatTensor(
            self.targets[index]
        )


# Initialize experiment data
experiment_data = {
    "batch_size_tuning": {},
}

# Hyperparameter tuning for different batch sizes
batch_sizes = [16, 32, 64, 128]
for batch_size in batch_sizes:
    # Create dataset and split into train/val
    dataset = CosmologyDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Define a simple neural network model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.fc(x)

    # Initialize model, criterion, optimizer
    model = SimpleModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare storage for experiment results for this batch size
    experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
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

        experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"]["losses"][
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
                experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"][
                    "predictions"
                ].extend(outputs.cpu().numpy())
                experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"][
                    "ground_truth"
                ].extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"]["losses"][
            "val"
        ].append(val_loss)

        # Compute Bayesian Evidence Ratio (BER) as a placeholder value
        BER = np.exp(-val_loss)
        experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"]["metrics"][
            "val"
        ].append(BER)

        print(
            f"Batch size {batch_size}, Epoch {epoch + 1}: validation_loss = {val_loss:.4f}, BER = {BER:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
