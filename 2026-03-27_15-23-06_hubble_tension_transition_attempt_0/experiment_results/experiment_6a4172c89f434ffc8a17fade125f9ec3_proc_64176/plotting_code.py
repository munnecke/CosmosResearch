import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Extract data for plotting
optimizer_data = experiment_data["optimizer_ablation"]["synthetic_cosmology"]
losses = optimizer_data["losses"]
predictions = optimizer_data["predictions"]
ground_truth = optimizer_data["ground_truth"]
optimizers = optimizer_data["optimizers"]
metrics = optimizer_data["metrics"]

# Plot training and validation losses
for idx, optimizer in enumerate(optimizers):
    try:
        plt.figure()
        epochs = np.arange(1, len(losses["train"]) // len(optimizers) + 1)
        train_losses = losses["train"][idx * len(epochs) : (idx + 1) * len(epochs)]
        val_losses = losses["val"][idx * len(epochs) : (idx + 1) * len(epochs)]
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"{optimizer} Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{optimizer}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for optimizer {optimizer}: {e}")
        plt.close()

# Plot BER over epochs for each optimizer
for idx, optimizer in enumerate(optimizers):
    try:
        plt.figure()
        epochs = np.arange(1, len(metrics["val"]) // len(optimizers) + 1)
        ber_vals = metrics["val"][idx * len(epochs) : (idx + 1) * len(epochs)]
        plt.plot(epochs, ber_vals, label="BER")
        plt.title(f"{optimizer} BER Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Bayesian Evidence Ratio (BER)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{optimizer}_BER.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating BER plot for optimizer {optimizer}: {e}")
        plt.close()

# Plot ground truth vs. predictions
try:
    plt.figure()
    scatter_idx = np.random.choice(range(len(ground_truth)), size=100, replace=False)
    plt.scatter(
        ground_truth[scatter_idx], np.array(predictions)[scatter_idx], alpha=0.5
    )
    plt.plot([-0.5, 2.5], [-0.5, 2.5], "r", linestyle="--")
    plt.title("Predictions vs Ground Truth (Sampled)")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(os.path.join(working_dir, "predictions_vs_ground_truth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
