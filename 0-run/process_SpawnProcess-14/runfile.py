import os
import numpy as np

# Load experiment data from numpy file
working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Iterate over activation functions' results
for experiment_key, data in experiment_data.items():
    print(f"Dataset: {experiment_key}")

    # Extract and print the final training loss
    train_loss = data["losses"]["train"][-1]
    print(f"Final training loss: {train_loss:.4f}")

    # Extract and print the final validation loss
    val_loss = data["losses"]["val"][-1]
    print(f"Final validation loss: {val_loss:.4f}")

    # Extract and print the final BER (validation metric)
    val_ber = data["metrics"]["val"][-1]
    print(f"Final validation BER: {val_ber:.4f}")

    print()  # Add a blank line for better separation
