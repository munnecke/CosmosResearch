import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses
try:
    plt.figure()
    epochs = range(
        1, len(experiment_data["synthetic_cosmology"]["losses"]["train"]) + 1
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_cosmology"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_cosmology"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_cosmology_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot BER metric over epochs
try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["synthetic_cosmology"]["metrics"]["val"],
        label="Validation BER",
    )
    plt.title("Bayesian Evidence Ratio (BER) over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("BER")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_cosmology_ber.png"))
    plt.close()
except Exception as e:
    print(f"Error creating BER plot: {e}")
    plt.close()

# Scatter plot for predictions vs ground truth
try:
    plt.figure()
    predictions = experiment_data["synthetic_cosmology"]["predictions"]
    ground_truth = experiment_data["synthetic_cosmology"]["ground_truth"]
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.title("Predictions vs. Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(working_dir, "synthetic_cosmology_predictions_vs_ground_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
