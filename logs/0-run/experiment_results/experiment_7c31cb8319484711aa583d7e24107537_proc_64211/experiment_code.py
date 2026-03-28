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


# Create dataset and split into train/val
dataset = CosmologyDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Experiment data container
experiment_data = {
    "regularization_ablation": {
        "L1": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "L2": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "Dropout": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}


# Function to create model
def create_model(hidden_layer_size, use_dropout=False):
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            layers = [
                nn.Linear(1, hidden_layer_size),
                nn.ReLU(),
                nn.Dropout(0.5) if use_dropout else nn.Identity(),
                nn.Linear(hidden_layer_size, hidden_layer_size),
                nn.ReLU(),
                nn.Dropout(0.5) if use_dropout else nn.Identity(),
                nn.Linear(hidden_layer_size, 1),
            ]
            self.fc = nn.Sequential(*layers)

        def forward(self, x):
            return self.fc(x)

    return SimpleModel().to(device)


# Regularization experiments
regularization_types = {
    "L1": {"weight_decay": 0, "l1_lambda": 0.01, "use_dropout": False},
    "L2": {"weight_decay": 0.01, "l1_lambda": 0, "use_dropout": False},
    "Dropout": {"weight_decay": 0, "l1_lambda": 0, "use_dropout": True},
}

for regularization, params in regularization_types.items():
    model = create_model(128, use_dropout=params.get("use_dropout", False))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=0.001, weight_decay=params["weight_decay"]
    )

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
            if params["l1_lambda"] > 0:
                l1_regularization = params["l1_lambda"] * sum(
                    param.abs().sum() for param in model.parameters()
                )
                loss += l1_regularization
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        experiment_data["regularization_ablation"][regularization]["losses"][
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
                experiment_data["regularization_ablation"][regularization][
                    "predictions"
                ].extend(outputs.cpu().numpy())
                experiment_data["regularization_ablation"][regularization][
                    "ground_truth"
                ].extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        experiment_data["regularization_ablation"][regularization]["losses"][
            "val"
        ].append(val_loss)

        # Compute Bayesian Evidence Ratio (BER) as a placeholder value
        BER = np.exp(-val_loss)
        experiment_data["regularization_ablation"][regularization]["metrics"][
            "val"
        ].append(BER)

        print(
            f"Epoch {epoch + 1}: Regularization = {regularization}, Validation Loss = {val_loss:.4f}, BER = {BER:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
