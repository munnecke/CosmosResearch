import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()
data = experiment_data["learning_rate_ablation"]["synthetic_cosmology"]

try:
    for i, lr in enumerate(data["learning_rates"]):
        # Plot training and validation loss curves
        plt.figure()
        epochs = range(
            1, len(data["losses"]["train"]) // len(data["learning_rates"]) + 1
        )
        subset_train_loss = data["losses"]["train"][i :: len(data["learning_rates"])]
        subset_val_loss = data["losses"]["val"][i :: len(data["learning_rates"])]
        plt.plot(epochs, subset_train_loss, "b-", label="Training Loss")
        plt.plot(epochs, subset_val_loss, "r-", label="Validation Loss")
        plt.title(f"Loss Curves for LR = {lr}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_lr_{lr}.png"))
        plt.close()
except Exception as e:
    print(f"Error plotting loss curves: {e}")
    plt.close()

try:
    # Scatter plot for predictions vs ground truth
    plt.figure()
    predictions = np.array(data["predictions"])
    ground_truth = np.array(data["ground_truth"])
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Predictions vs Ground Truth")
    plt.savefig(os.path.join(working_dir, f"predictions_vs_ground_truth.png"))
    plt.close()
except Exception as e:
    print(f"Error plotting predictions vs ground truth: {e}")
    plt.close()
