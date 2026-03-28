import os
import numpy as np

# Load the experiment data
working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Extract and print metrics for each dataset
for dataset_name, dataset_info in experiment_data.items():
    print(f"Dataset: {dataset_name}")
    if "metrics" in dataset_info:
        metrics = dataset_info["metrics"]
        if "train" in metrics:
            train_metrics = metrics["train"]
            # Print the final or best training BER
            if train_metrics:
                final_train_BER = train_metrics[-1]
                print(f"Final training BER: {final_train_BER:.4f}")
