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


# Define a function to plot the losses
def plot_losses(training_losses, validation_losses, regularization_type):
    try:
        plt.figure()
        epochs = range(1, len(training_losses) + 1)
        plt.plot(epochs, training_losses, "b", label="Training loss")
        plt.plot(epochs, validation_losses, "r", label="Validation loss")
        plt.title(f"{regularization_type} Regularization: Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{regularization_type}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {regularization_type} losses: {e}")
        plt.close()


# Iterate over each regularization type and plot the losses
for regularization_type, data in experiment_data["regularization_ablation"].items():
    plot_losses(data["losses"]["train"], data["losses"]["val"], regularization_type)
