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

optimizers = experiment_data["optimizer_ablation"]["synthetic_cosmology"]["optimizers"]
train_losses = experiment_data["optimizer_ablation"]["synthetic_cosmology"]["losses"][
    "train"
]
val_losses = experiment_data["optimizer_ablation"]["synthetic_cosmology"]["losses"][
    "val"
]
ber_values = experiment_data["optimizer_ablation"]["synthetic_cosmology"]["metrics"][
    "val"
]
epochs = len(val_losses[0])

for i, optimizer in enumerate(optimizers):
    try:
        plt.figure()
        plt.plot(range(epochs), train_losses[i], label="Train Loss")
        plt.plot(range(epochs), val_losses[i], label="Validation Loss")
        plt.title(f"{optimizer} - Training/Validation Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{optimizer}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {optimizer}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(range(epochs), ber_values[i], label="Bayesian Evidence Ratio")
        plt.title(f"{optimizer} - BER over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("BER")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{optimizer}_BER.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating BER plot for {optimizer}: {e}")
        plt.close()
