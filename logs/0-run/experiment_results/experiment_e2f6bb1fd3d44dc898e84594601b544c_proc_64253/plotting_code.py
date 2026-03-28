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

# Plot training and validation losses for each variation
for variation in experiment_data["output_layer_variation"]["synthetic_cosmology"][
    "variation_types"
]:
    try:
        plt.figure()
        losses_train = experiment_data["output_layer_variation"]["synthetic_cosmology"][
            "losses"
        ]["train"]
        losses_val = experiment_data["output_layer_variation"]["synthetic_cosmology"][
            "losses"
        ]["val"]
        epochs = np.arange(1, len(losses_train) + 1)
        plt.plot(epochs, losses_train, label="Training Loss")
        plt.plot(epochs, losses_val, label="Validation Loss")
        plt.title(f"Loss Curves - {variation}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{variation}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {variation}: {e}")
        plt.close()

# Plot predictions vs ground truth for the 'extra_layer' variation
try:
    predictions = experiment_data["output_layer_variation"]["synthetic_cosmology"][
        "predictions"
    ]
    ground_truth = experiment_data["output_layer_variation"]["synthetic_cosmology"][
        "ground_truth"
    ]
    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.plot(ground_truth, ground_truth, "r--", label="Ground Truth Line")
    plt.title(f"Predictions vs Ground Truth - Extra Layer")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"pred_vs_gt_extra_layer.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predict vs ground truth plot for extra_layer: {e}")
    plt.close()
