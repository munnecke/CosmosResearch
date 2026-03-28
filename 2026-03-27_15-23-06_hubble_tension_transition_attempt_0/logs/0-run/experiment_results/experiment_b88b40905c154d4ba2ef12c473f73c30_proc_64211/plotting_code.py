import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data with exception handling
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Plot training and validation loss curves
    plt.figure()
    hidden_layer_sizes = experiment_data["loss_function_ablation"][
        "synthetic_cosmology"
    ]["hidden_layer_sizes"]
    for idx, size in enumerate(hidden_layer_sizes):
        train_losses = experiment_data["loss_function_ablation"]["synthetic_cosmology"][
            "losses"
        ]["train"][idx * 20 : (idx + 1) * 20]
        val_losses = experiment_data["loss_function_ablation"]["synthetic_cosmology"][
            "losses"
        ]["val"][idx * 20 : (idx + 1) * 20]
        plt.plot(train_losses, label=f"Train Loss (size={size})")
        plt.plot(val_losses, label=f"Val Loss (size={size})")

    plt.title("Training and Validation Loss Curves\nSynthetic Cosmology Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_cosmology_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    # Plot validation BER over epochs
    plt.figure()
    for idx, size in enumerate(hidden_layer_sizes):
        val_BER = experiment_data["loss_function_ablation"]["synthetic_cosmology"][
            "metrics"
        ]["val"][idx * 20 : (idx + 1) * 20]
        plt.plot(val_BER, label=f"Val BER (size={size})")

    plt.title(
        "Validation Bayesian Evidence Ratio (BER) Over Epochs\nSynthetic Cosmology Dataset"
    )
    plt.xlabel("Epoch")
    plt.ylabel("BER")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_cosmology_BER_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating BER plot: {e}")
    plt.close()

try:
    # Plot predictions versus ground truth
    plt.figure()
    predictions = experiment_data["loss_function_ablation"]["synthetic_cosmology"][
        "predictions"
    ]
    ground_truth = experiment_data["loss_function_ablation"]["synthetic_cosmology"][
        "ground_truth"
    ]
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.title("Predictions vs Ground Truth\nSynthetic Cosmology Dataset")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(working_dir, "synthetic_cosmology_predictions_vs_ground_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
