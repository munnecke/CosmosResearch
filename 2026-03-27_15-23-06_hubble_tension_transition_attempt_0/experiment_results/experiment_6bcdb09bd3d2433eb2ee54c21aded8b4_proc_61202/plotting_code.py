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

try:
    # Training and validation losses for model with batch normalization
    plt.figure()
    train_losses = experiment_data["batch_normalization"]["synthetic_cosmology"][
        "losses"
    ]["train"]
    val_losses = experiment_data["batch_normalization"]["synthetic_cosmology"][
        "losses"
    ]["val"]
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curves with Batch Normalization")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_with_batch_norm.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot with batch norm: {e}")
    plt.close()

try:
    # Predictions vs Ground Truth with Batch Norm
    plt.figure()
    predictions = experiment_data["batch_normalization"]["synthetic_cosmology"][
        "predictions"
    ]
    ground_truth = experiment_data["batch_normalization"]["synthetic_cosmology"][
        "ground_truth"
    ]
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.plot(
        [min(ground_truth), max(ground_truth)],
        [min(ground_truth), max(ground_truth)],
        "r--",
    )
    plt.title("Predictions vs Ground Truth with Batch Normalization")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(os.path.join(working_dir, "pred_vs_gt_with_batch_norm.png"))
    plt.close()
except Exception as e:
    print(f"Error creating prediction plot with batch norm: {e}")
    plt.close()

# Repeat similar steps for models without batch normalization if needed
