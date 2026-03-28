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


# Define function to plot data
def plot_dropout_rate_data(dropout_rate, losses, metrics, predictions, ground_truth):
    # Plot training/validation loss curves
    try:
        plt.figure()
        plt.plot(losses["train"], label="Train Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title(f"Loss Curves (Dropout: {dropout_rate})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"loss_curves_dropout_{dropout_rate}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for dropout {dropout_rate}: {e}")
        plt.close()

    # Plot BER over epochs
    try:
        plt.figure()
        plt.plot(metrics["val"], label="Validation BER")
        plt.title(f"BER over Epochs (Dropout: {dropout_rate})")
        plt.xlabel("Epoch")
        plt.ylabel("BER")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"ber_epochs_dropout_{dropout_rate}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating BER plot for dropout {dropout_rate}: {e}")
        plt.close()

    # Scatter plot of predictions against ground truth
    try:
        plt.figure()
        plt.scatter(ground_truth, predictions, alpha=0.5)
        plt.title(f"Predictions vs Ground Truth (Dropout: {dropout_rate})")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(
                working_dir, f"predictions_vs_ground_truth_dropout_{dropout_rate}.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot for dropout {dropout_rate}: {e}")
        plt.close()


# Generate plots for each dropout rate
experiment = experiment_data["dropout_rate_tuning"]["synthetic_cosmology"]
for dropout_rate, losses, metrics, predictions, ground_truth in zip(
    [0.1, 0.3, 0.5],
    experiment["losses"]["train"],
    experiment["metrics"]["val"],
    experiment["predictions"],
    experiment["ground_truth"],
):
    plot_dropout_rate_data(dropout_rate, losses, metrics, predictions, ground_truth)
