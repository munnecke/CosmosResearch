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

datasets = ["sinusoidal", "linear", "quadratic"]
num_epochs = 20
plot_interval = max(num_epochs // 5, 1)

for dataset_name in datasets:
    # Plot train and validation loss curves
    try:
        plt.figure()
        epochs = range(1, num_epochs + 1)
        train_losses = experiment_data["dataset_variation_ablation"][dataset_name][
            "losses"
        ]["train"]
        val_losses = experiment_data["dataset_variation_ablation"][dataset_name][
            "losses"
        ]["val"]
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{dataset_name.capitalize()} Dataset: Loss Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss curves for {dataset_name}: {e}")
        plt.close()
