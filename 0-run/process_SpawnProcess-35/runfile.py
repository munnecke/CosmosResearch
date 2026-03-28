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


# Define enhanced synthetic dataset with additional features
class CosmologyDataset(Dataset):
    def __init__(self, size=1000, poly_degree=3, noise_std=0.1):
        base_data = np.random.uniform(0, 2, (size, 1))  # Simulated redshifts
        poly_features = np.hstack([base_data**d for d in range(1, poly_degree + 1)])
        noise_features = noise_std * np.random.randn(size, poly_degree)
        self.data = np.hstack([poly_features, noise_features])
        # Simulated cosmological values, e.g., H0(z), w(z)
        self.targets = np.sin(base_data) + 0.1 * np.random.randn(size, 1)

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
    "ablation_input_dimension": {
        "synthetic_cosmology": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "input_dimensions": [],
        },
    }
}


# Function to create model considering new input dimensions
def create_model(input_size, hidden_layer_size):
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_size, hidden_layer_size),
                nn.ReLU(),
                nn.Linear(hidden_layer_size, hidden_layer_size),
                nn.ReLU(),
                nn.Linear(hidden_layer_size, 1),
            )

        def forward(self, x):
            return self.fc(x)

    return SimpleModel().to(device)


input_dimensions = [1, 2, 4, 6, 8]
hidden_layer_size = 64
for input_dim in input_dimensions:
    model = create_model(input_dim, hidden_layer_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 20
    predictions = []
    ground_truth = []

    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs[:, :input_dim].to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        experiment_data["ablation_input_dimension"]["synthetic_cosmology"]["losses"][
            "train"
        ].append(training_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs[:, :input_dim].to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                predictions.extend(outputs.cpu().numpy())
                ground_truth.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        experiment_data["ablation_input_dimension"]["synthetic_cosmology"]["losses"][
            "val"
        ].append(val_loss)

        # Compute Bayesian Evidence Ratio (BER) as a placeholder value
        BER = np.exp(-val_loss)
        experiment_data["ablation_input_dimension"]["synthetic_cosmology"]["metrics"][
            "val"
        ].append(BER)

        print(
            f"Epoch {epoch + 1}: Input Dim = {input_dim}, Validation Loss = {val_loss:.4f}, BER = {BER:.4f}"
        )

    experiment_data["ablation_input_dimension"]["synthetic_cosmology"][
        "predictions"
    ].append(predictions)
    experiment_data["ablation_input_dimension"]["synthetic_cosmology"][
        "ground_truth"
    ].append(ground_truth)
    experiment_data["ablation_input_dimension"]["synthetic_cosmology"][
        "input_dimensions"
    ].append(input_dim)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
