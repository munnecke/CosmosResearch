import os
import numpy as np

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Process and print metrics
for dataset_name, dataset_info in experiment_data.items():
    print(f"Dataset: {dataset_name}")

    # Extract metrics
    metrics = dataset_info["metrics"]
    losses = dataset_info["losses"]

    # Print final validation losses
    if "val" in losses:
        final_val_loss = losses["val"][-1]
        print(f"Final Validation Loss: {final_val_loss:.4f}")

    # Print final BER metric
    if "val" in metrics:
        final_BER = metrics["val"][-1]
        print(f"Final Bayesian Evidence Ratio (BER): {final_BER:.4f}")
