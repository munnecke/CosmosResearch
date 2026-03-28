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
        self.data = np.random.uniform(0, 2, (size, 1))
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


# Define a custom Swish activation as a Module
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self, activation_fn):
        super(SimpleModel, self).__init__()
        if activation_fn == "relu":
            activation_layer = nn.ReLU()
        elif activation_fn == "leaky_relu":
            activation_layer = nn.LeakyReLU()
        elif activation_fn == "elu":
            activation_layer = nn.ELU()
        elif activation_fn == "swish":
            activation_layer = Swish()
        else:
            raise ValueError("Unsupported activation function")

        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            activation_layer,
            nn.Linear(64, 64),
            activation_layer,
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


# Setup hyperparameter tuning
activation_functions = ["relu", "leaky_relu", "elu", "swish"]
experiment_data = {
    f"activation_{func}": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for func in activation_functions
}

# Training loop for each activation function
learning_rate = 0.001
num_epochs = 20
for activation_fn in activation_functions:
    model = SimpleModel(activation_fn).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    experiment_key = f"activation_{activation_fn}"

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

        # Record training loss
        experiment_data[experiment_key]["losses"]["train"].append(
            training_loss / len(train_loader)
        )

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

                experiment_data[experiment_key]["predictions"].extend(
                    outputs.cpu().numpy()
                )
                experiment_data[experiment_key]["ground_truth"].extend(
                    targets.cpu().numpy()
                )

        val_loss /= len(val_loader)
        experiment_data[experiment_key]["losses"]["val"].append(val_loss)

        BER = np.exp(-val_loss)
        experiment_data[experiment_key]["metrics"]["val"].append(BER)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Activation {activation_fn}: Last validation_loss = {val_loss:.4f}, BER = {BER:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
