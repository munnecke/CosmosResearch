import os
import numpy as np

# Get the working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load the experiment data
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()


# Extract and print metrics
def extract_metrics():
    for analysis_type in experiment_data.keys():
        for dataset_name, dataset_data in experiment_data[analysis_type].items():
            print(f"Dataset: {dataset_name}")

            # Extract losses and metrics
            train_losses = dataset_data["losses"]["train"]
            val_losses = dataset_data["losses"]["val"]
            val_metrics = dataset_data["metrics"]["val"]

            # Print final training loss
            final_train_loss = train_losses[-1]
            print(f"Final Training Loss: {final_train_loss}")

            # Print final validation loss
            final_val_loss = val_losses[-1]
            print(f"Final Validation Loss: {final_val_loss}")

            # Print best validation BER (smallest value should be the 'best')
            best_val_BER = min(val_metrics)
            print(f"Best Validation BER: {best_val_BER}")


extract_metrics()
