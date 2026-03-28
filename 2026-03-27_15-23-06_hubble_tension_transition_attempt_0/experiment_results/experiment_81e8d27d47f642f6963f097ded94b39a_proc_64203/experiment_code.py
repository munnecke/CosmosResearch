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

# Experiment data container
experiment_data = {"data_noise_level_ablation": {}}


# Define synthetic dataset with noise multiplier
class CosmologyDataset(Dataset):
    def __init__(self, size=1000, noise_multiplier=0.1):
        self.data = np.random.uniform(0, 2, (size, 1))  # Simulated redshifts
        # Simulated cosmological values, e.g., H0(z), w(z) with varying noise
        self.targets = np.sin(self.data) + noise_multiplier * np.random.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), torch.FloatTensor(
            self.targets[index]
        )


# Function to create model with fixed hidden layer size
def create_model(hidden_layer_size=64):
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


# Noise levels to test
noise_levels = [0.05, 0.1, 0.15, 0.2]
for noise in noise_levels:
    # Create dataset and split into train/val
    dataset = CosmologyDataset(noise_multiplier=noise)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize storage for the current noise level
    experiment_data["data_noise_level_ablation"][f"noise_{noise}"] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Create model
    model = create_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

        experiment_data["data_noise_level_ablation"][f"noise_{noise}"]["losses"][
            "train"
        ].append(training_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        predictions, ground_truth = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Collect predictions and ground truth
                predictions.extend(outputs.cpu().numpy())
                ground_truth.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        experiment_data["data_noise_level_ablation"][f"noise_{noise}"]["losses"][
            "val"
        ].append(val_loss)
        experiment_data["data_noise_level_ablation"][f"noise_{noise}"][
            "predictions"
        ] = predictions
        experiment_data["data_noise_level_ablation"][f"noise_{noise}"][
            "ground_truth"
        ] = ground_truth

        # Compute Bayesian Evidence Ratio (BER) as a placeholder value
        BER = np.exp(-val_loss)
        experiment_data["data_noise_level_ablation"][f"noise_{noise}"]["metrics"][
            "val"
        ].append(BER)

        print(
            f"Epoch {epoch + 1}: Noise = {noise}, Validation Loss = {val_loss:.4f}, BER = {BER:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
