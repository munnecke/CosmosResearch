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
    epochs = range(1, 21)
    for idx, size in enumerate(
        experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_cosmology"][
            "hidden_layer_sizes"
        ]
    ):
        train_losses = experiment_data["hyperparam_tuning_hidden_layer_size"][
            "synthetic_cosmology"
        ]["losses"]["train"][idx * 20 : (idx + 1) * 20]
        val_losses = experiment_data["hyperparam_tuning_hidden_layer_size"][
            "synthetic_cosmology"
        ]["losses"]["val"][idx * 20 : (idx + 1) * 20]
        plt.plot(epochs, train_losses, label=f"Train Loss Size {size}")
        plt.plot(epochs, val_losses, label=f"Val Loss Size {size}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_cosmology_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot BER
try:
    plt.figure()
    for idx, size in enumerate(
        experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_cosmology"][
            "hidden_layer_sizes"
        ]
    ):
        ber_vals = experiment_data["hyperparam_tuning_hidden_layer_size"][
            "synthetic_cosmology"
        ]["metrics"]["val"][idx * 20 : (idx + 1) * 20]
        plt.plot(epochs, ber_vals, label=f"BER Size {size}")
    plt.xlabel("Epochs")
    plt.ylabel("BER")
    plt.title("Bayesian Evidence Ratio per Epoch")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_cosmology_BER_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating BER plot: {e}")
    plt.close()

# Plot predictions vs. ground truth
try:
    plt.figure()
    predictions = experiment_data["hyperparam_tuning_hidden_layer_size"][
        "synthetic_cosmology"
    ]["predictions"]
    ground_truth = experiment_data["hyperparam_tuning_hidden_layer_size"][
        "synthetic_cosmology"
    ]["ground_truth"]
    plt.scatter(ground_truth, predictions, alpha=0.2)
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Predictions vs. Ground Truth")
    plt.savefig(
        os.path.join(working_dir, "synthetic_cosmology_predictions_vs_ground_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
