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
    # Training and Validation Loss Plot
    plt.figure()
    train_losses = experiment_data["learning_rate_scheduler_ablation"][
        "synthetic_cosmology"
    ]["losses"]["train"]
    val_losses = experiment_data["learning_rate_scheduler_ablation"][
        "synthetic_cosmology"
    ]["losses"]["val"]
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curve_synthetic_cosmology.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # BER Plot
    plt.figure()
    ber = experiment_data["learning_rate_scheduler_ablation"]["synthetic_cosmology"][
        "metrics"
    ]["val"]
    plt.plot(ber, label="BER")
    plt.title("BER over Epochs (Synthetic Cosmology)")
    plt.xlabel("Epoch")
    plt.ylabel("BER")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "ber_curve_synthetic_cosmology.png"))
    plt.close()
except Exception as e:
    print(f"Error creating BER plot: {e}")
    plt.close()

try:
    # Predictions vs Ground Truth Plot
    plt.figure()
    predictions = experiment_data["learning_rate_scheduler_ablation"][
        "synthetic_cosmology"
    ]["predictions"]
    ground_truth = experiment_data["learning_rate_scheduler_ablation"][
        "synthetic_cosmology"
    ]["ground_truth"]
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.title("Predictions vs Ground Truth (Synthetic Cosmology)")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(working_dir, "predictions_vs_ground_truth_synthetic_cosmology.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
