import os
import numpy as np

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Extract metrics and print them clearly
for dataset_name, dataset_data in experiment_data.items():
    print(f"Dataset: {dataset_name}")

    # Metrics extraction
    for metric_name, values in dataset_data["metrics"].items():
        if values:
            final_value = values[-1]  # Assuming we are interested in the final value
            explicit_metric_name = (
                "train BER" if metric_name == "train" else f"{metric_name} metric"
            )
            print(f"{explicit_metric_name}: {final_value}")
