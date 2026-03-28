import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets from Hugging Face
datasets = {
    "numerai": load_dataset("numerai/numerai_dataset", split="train"),
    "kaggle_chaii": load_dataset("kaggle-chaii-wip/koi", split="train[:300]"),
    "oscar": load_dataset("oscar-corpus/OSCAR-2201", split="train[:300]"),
}


# Preprocessing and defining datasets
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def preprocess_data(dataset, input_columns, target_column):
    X = dataset[input_columns]
    y = dataset[target_column]
    scaler = MinMaxScaler()
    X = np.array([np.array(item) for item in X])
    y = np.array([item for item in y])
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


datasets_preprocessed = {}
for name, dataset in datasets.items():
    if name == "numerai":
        X_train, X_test, y_train, y_test = preprocess_data(
            dataset, ["features"], "targets"
        )
    elif name == "kaggle_chaii":
        X_train, X_test, y_train, y_test = preprocess_data(
            dataset, ["document"], "question"
        )
    elif name == "oscar":
        X_train, X_test, y_train, y_test = preprocess_data(dataset, ["text"], "text")
    datasets_preprocessed[name] = {
        "train": CustomDataset(X_train, y_train),
        "val": CustomDataset(X_test, y_test),
    }

# Experiment data container
experiment_data = {
    "models": {
        name: {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
        for name in datasets_preprocessed
    }
}


# Model definition
class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


# Training and validation function
def train_and_evaluate(data, name):
    train_loader = DataLoader(data["train"], batch_size=32, shuffle=True)
    val_loader = DataLoader(data["val"], batch_size=32)

    model = SimpleModel(data["train"][0][0].shape[0]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        experiment_data["models"][name]["losses"]["train"].append(
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
                loss = criterion(outputs, targets.unsqueeze(1))
                val_loss += loss.item()

                # Record predictions
                experiment_data["models"][name]["predictions"].extend(
                    outputs.cpu().numpy()
                )
                experiment_data["models"][name]["ground_truth"].extend(
                    targets.cpu().numpy()
                )

        val_loss /= len(val_loader)
        experiment_data["models"][name]["losses"]["val"].append(val_loss)

        # Calculate Bayesian Evidence Ratio placeholder
        BER = np.exp(-val_loss)
        experiment_data["models"][name]["metrics"]["val"].append(BER)

        print(
            f"Epoch {epoch + 1} [{name}]: Validation Loss = {val_loss:.4f}, BER = {BER:.4f}"
        )


# Train and evaluate on each dataset
for name, data in datasets_preprocessed.items():
    train_and_evaluate(data, name)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
