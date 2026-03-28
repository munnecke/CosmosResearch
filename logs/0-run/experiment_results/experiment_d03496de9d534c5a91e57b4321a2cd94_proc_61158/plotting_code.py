import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load the experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Create visualizations for each batch size
for batch_size_key, data in experiment_data["batch_size_tuning"].items():
    try:
        # Plot the losses
        plt.figure()
        epochs = range(1, len(data["losses"]["train"]) + 1)
        plt.plot(epochs, data["losses"]["train"], label="Train Loss")
        plt.plot(epochs, data["losses"]["val"], label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves for {batch_size_key}")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{batch_size_key}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {batch_size_key}: {e}")
        plt.close()

    try:
        # Plot the BER metric
        plt.figure()
        plt.plot(epochs, data["metrics"]["val"], label="Validation BER")
        plt.xlabel("Epochs")
        plt.ylabel("BER")
        plt.title(f"BER Metric for {batch_size_key}")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{batch_size_key}_BER_metric.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating BER plot for {batch_size_key}: {e}")
        plt.close()

    try:
        # Plot predictions vs ground truth
        plt.figure()
        plt.scatter(data["ground_truth"], data["predictions"], alpha=0.5)
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.title(f"Predictions vs Ground Truth for {batch_size_key}")
        plt.savefig(
            os.path.join(
                working_dir, f"{batch_size_key}_predictions_vs_ground_truth.png"
            )
        )
        plt.close()
    except Exception as e:
        print(
            f"Error creating predictions vs ground truth plot for {batch_size_key}: {e}"
        )
        plt.close()
