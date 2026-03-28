import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# Create dataset and split into train/val
dataset = CosmologyDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# Define a simple neural network model with configurable initialization
class SimpleModel(nn.Module):
    def __init__(self, init_method=None):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        if init_method:
            self._initialize_weights(init_method)

    def forward(self, x):
        return self.fc(x)

    def _initialize_weights(self, init_method):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_uniform_(layer.weight)
                elif init_method == "he":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")


# Define hyperparameter tuning experiments
init_methods = ["default", "xavier", "he"]
experiment_data = {
    f"init_{method}": {
        "synthetic_cosmology": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
    for method in init_methods
}

num_epochs = 20

# Training loop with different initialization methods
for init_method in init_methods:
    model = SimpleModel(
        init_method=(init_method if init_method != "default" else None)
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

        experiment_data[f"init_{init_method}"]["synthetic_cosmology"]["losses"][
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
                experiment_data[f"init_{init_method}"]["synthetic_cosmology"][
                    "predictions"
                ].extend(outputs.cpu().numpy())
                experiment_data[f"init_{init_method}"]["synthetic_cosmology"][
                    "ground_truth"
                ].extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        experiment_data[f"init_{init_method}"]["synthetic_cosmology"]["losses"][
            "val"
        ].append(val_loss)

        # Simplified Bayesian Evidence Ratio placeholder
        BER = np.exp(-val_loss)
        experiment_data[f"init_{init_method}"]["synthetic_cosmology"]["metrics"][
            "val"
        ].append(BER)

        print(
            f"Init {init_method.capitalize()} - Epoch {epoch + 1}: validation_loss = {val_loss:.4f}, BER = {BER:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
