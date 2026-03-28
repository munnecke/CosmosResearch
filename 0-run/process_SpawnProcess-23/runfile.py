import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from datasets import load_dataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Import datasets
datasets = {
    "glue_stsb": load_dataset("glue", "stsb", split="train").select(range(1000)),
    "math_qa": load_dataset("math_qa", split="train").select(range(1000)),
    "trec": load_dataset("trec", split="train").select(range(1000)),
}


# Define dataset class for Hugging Face datasets
class HFDataset(Dataset):
    def __init__(self, dataset, feature_column, target_column, transform=None):
        self.dataset = dataset
        self.feature_column = feature_column
        self.target_column = target_column
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features = self.dataset[idx][self.feature_column]
        target = self.dataset[idx][self.target_column]
        if self.transform:
            features, target = self.transform(features, target)
        return torch.FloatTensor(features), torch.FloatTensor([target])


# Transformation to normalize the dataset (placeholder)
def transform_fn(features, target):
    return (
        np.asarray(features).astype(np.float32) / 1000.0,
        target / 100.0,
    )  # Normalize features and target


# Prepare datasets
experiment_data = {
    "glue_stsb": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "math_qa": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "trec": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}


def create_loader(dataset_name, feature_column, target_column, transform_fn):
    dataset = HFDataset(
        datasets[dataset_name], feature_column, target_column, transform=transform_fn
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    return train_loader, val_loader


train_loaders = {}
val_loaders = {}

# Set feature_column and target_column according to the dataset structure
train_loaders["glue_stsb"], val_loaders["glue_stsb"] = create_loader(
    "glue_stsb", "sentence1", "label", transform_fn
)
train_loaders["math_qa"], val_loaders["math_qa"] = create_loader(
    "math_qa", "question", "correct", transform_fn
)
train_loaders["trec"], val_loaders["trec"] = create_loader(
    "trec", "text", "label-coarse", transform_fn
)


# Create model
def create_model():
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.fc(x)

    return SimpleModel().to(device)


model = create_model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_and_evaluate_model(dataset_name, train_loader, val_loader):

    num_epochs = 20
    for epoch in range(num_epochs):
        # Training
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

        experiment_data[dataset_name]["losses"]["train"].append(
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
            f"Dataset {dataset_name}: Epoch {epoch + 1}, Validation Loss = {val_loss:.4f}, BER = {BER:.4f}"
        )


# Run training and evaluation for each dataset
for dataset_name in datasets.keys():
    train_and_evaluate_model(
        dataset_name, train_loaders[dataset_name], val_loaders[dataset_name]
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
