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

init_methods = ["default", "xavier", "he"]
for init_method in init_methods:
    try:
        # Plot training and validation losses
        plt.figure()
        train_losses = experiment_data[f"init_{init_method}"]["synthetic_cosmology"][
            "losses"
        ]["train"]
        val_losses = experiment_data[f"init_{init_method}"]["synthetic_cosmology"][
            "losses"
        ]["val"]
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"Training and Validation Loss - {init_method.capitalize()} Initialization"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_plot_{init_method}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {init_method}: {e}")
        plt.close()

    try:
        # Plot predictions vs ground truth
        plt.figure()
        predictions = experiment_data[f"init_{init_method}"]["synthetic_cosmology"][
            "predictions"
        ]
        ground_truth = experiment_data[f"init_{init_method}"]["synthetic_cosmology"][
            "ground_truth"
        ]
        plt.scatter(ground_truth, predictions, alpha=0.5)
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.title(
            f"Predictions vs Ground Truth - {init_method.capitalize()} Initialization"
        )
        plt.savefig(os.path.join(working_dir, f"predictions_plot_{init_method}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for {init_method}: {e}")
        plt.close()
