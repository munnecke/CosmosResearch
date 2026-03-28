import os
import numpy as np

# Load the experiment data
working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Extract the dataset name and metrics
for exp_name, datasets in experiment_data.items():
    for dataset_name, dataset_info in datasets.items():
        print(f"Dataset Name: {dataset_name}")

        # Fetch and print the final validation Bayesian Evidence Ratio (BER)
        if "val" in dataset_info["metrics"]:
            final_val_ber = dataset_info["metrics"]["val"][-1]
            print(f"Final Validation Bayesian Evidence Ratio (BER): {final_val_ber}")

        # Fetch and print the final training loss
        if "train" in dataset_info["losses"]:
            final_train_loss = dataset_info["losses"]["train"][-1]
            print(f"Final Training Loss: {final_train_loss}")

        # Fetch and print the final validation loss
        if "val" in dataset_info["losses"]:
            final_val_loss = dataset_info["losses"]["val"][-1]
            print(f"Final Validation Loss: {final_val_loss}")
