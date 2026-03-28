import os
import numpy as np

# Load the experiment data from the npy file
working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(experiment_data_path, allow_pickle=True).item()


# Function to print metrics
def print_metrics():
    for dataset_name, dataset_info in experiment_data.items():
        print(f"Dataset: {dataset_name}")

        # Extract metrics
        train_metrics = dataset_info["metrics"]["train"]
        train_losses = dataset_info["losses"]["train"]
        val_losses = dataset_info["losses"]["val"]

        # Print best or final values
        if train_metrics:
            print(f"Final train BER: {train_metrics[-1]:.4f}")

        if train_losses:
            print(f"Final train loss: {train_losses[-1]:.4f}")

        if val_losses:
            print(f"Final validation loss: {val_losses[-1]:.4f}")


# Execute the function
print_metrics()
