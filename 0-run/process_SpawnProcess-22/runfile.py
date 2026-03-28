import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets from HuggingFace
datasets = {
    "super_glue": load_dataset("super_glue", "boolq", split="train"),
    "cosmos_qa": load_dataset("cosmos_qa", split="train"),
    "cnn_dailymail": load_dataset("cnn_dailymail", "3.0.0", split="train"),
}


# Preprocessing example, assuming 'text' and 'label' for simplicity
def preprocess_data(dataset, input_key="input", target_key="target", max_len=50):
    processed_data = []
    targets = []
    for entry in dataset:
        text_input = entry[input_key][:max_len]  # Assuming text input
        target = (
            entry[target_key] if target_key in entry else 0
        )  # Assuming numeric target
        processed_data.append(text_input)
        targets.append(target)
    return processed_data, targets


# Convert to tensor datasets
tensor_datasets = {}
for name, ds in datasets.items():
    inputs, targets = preprocess_data(ds, "text", "label")
    input_tensor = torch.tensor(inputs)
    target_tensor = torch.tensor(targets)
    tensor_datasets[name] = torch.utils.data.TensorDataset(input_tensor, target_tensor)

experiment_data = {
    dataset_name: {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for dataset_name in datasets.keys()
}


# Define simple NN model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size):
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


# Train and evaluate model on each dataset
hidden_layer_size = 128
for dataset_name, tensor_ds in tensor_datasets.items():
    input_size = len(tensor_ds[0][0])  # Assuming consistent input size
    # Splitting datasets
    train_size = int(0.8 * len(tensor_ds))
    val_size = len(tensor_ds) - train_size
    train_dataset, val_dataset = random_split(tensor_ds, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = SimpleModel(input_size, hidden_layer_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs, targets = batch[0], batch[1]

            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float().view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        experiment_data[dataset_name]["losses"]["train"].append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                inputs, targets = batch[0], batch[1]
                outputs = model(inputs.float())
                loss = criterion(outputs, targets.float().view(-1, 1))
                val_loss += loss.item()

                experiment_data[dataset_name]["predictions"].extend(
                    outputs.cpu().numpy()
                )
                experiment_data[dataset_name]["ground_truth"].extend(
                    targets.cpu().numpy()
                )

        val_loss /= len(val_loader)
        experiment_data[dataset_name]["losses"]["val"].append(val_loss)

        # Placeholder for Bayesian Evidence Ratio (BER)
        BER = np.exp(-val_loss)
        experiment_data[dataset_name]["metrics"]["val"].append(BER)

        print(
            f"{dataset_name} - Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, BER: {BER:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
