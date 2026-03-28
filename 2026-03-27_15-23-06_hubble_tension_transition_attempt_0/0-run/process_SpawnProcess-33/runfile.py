import os
import numpy as np

# Get the working directory
working_dir = os.path.join(os.getcwd(), "working")


def load_and_display_metrics():
    # Load the experiment data
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()

    # Extract the specific dataset within the experiment data
    datasets = experiment_data["ablation_input_dimension"]

    for dataset_name, data in datasets.items():
        print(f"Dataset: {dataset_name}")

        # Retrieve metrics
        val_metrics = data["metrics"]["val"]
        val_losses = data["losses"]["val"]

        # Print the final or best value for validation loss and BER
        if val_losses:
            final_val_loss = val_losses[-1]
            print(f"Final Validation Loss: {final_val_loss:.4f}")

        if val_metrics:
            final_BER = val_metrics[-1]
            print(f"Final Bayesian Evidence Ratio (BER): {final_BER:.4f}")


# Execute the function to load and display the metrics
load_and_display_metrics()
