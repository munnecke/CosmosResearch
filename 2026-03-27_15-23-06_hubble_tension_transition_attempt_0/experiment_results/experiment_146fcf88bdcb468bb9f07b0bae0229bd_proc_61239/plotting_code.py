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

activation_functions = ["relu", "leaky_relu", "elu", "swish"]

for activation_fn in activation_functions:
    exp_key = f"activation_{activation_fn}"

    try:
        plt.figure()
        train_losses = experiment_data[exp_key]["losses"]["train"]
        val_losses = experiment_data[exp_key]["losses"]["val"]
        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"Loss Curves for {exp_key}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{activation_fn}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {activation_fn}: {e}")
        plt.close()

    try:
        plt.figure()
        val_metrics = experiment_data[exp_key]["metrics"]["val"]

        plt.plot(epochs, val_metrics, label="BER")
        plt.title(f"Validation Metric (BER) for {exp_key}")
        plt.xlabel("Epochs")
        plt.ylabel("BER")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"val_metric_{activation_fn}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating BER plot for {activation_fn}: {e}")
        plt.close()
