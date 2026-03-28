import os
import numpy as np

# Define the working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load the experiment data
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Extract and print metrics for the 'hubble_tension' dataset
dataset_name = "hubble_tension"
print(f"Dataset: {dataset_name}")

# Get and print the final BER-like metric for training
train_ber_metric = experiment_data[dataset_name]["metrics"]["train"][-1]
print(f"Final BER-like metric (train): {train_ber_metric}")

# Get and print the final training loss
train_loss = experiment_data[dataset_name]["losses"]["train"][-1]
print(f"Final training loss: {train_loss}")

# Get and print the final validation loss
val_loss = experiment_data[dataset_name]["losses"]["val"][-1]
print(f"Final validation loss: {val_loss}")
