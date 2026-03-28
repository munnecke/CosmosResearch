import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Extract data from experiment data
weight_decay_data = experiment_data["weight_decay_tuning"]["synthetic_cosmology"]

# Plot training and validation losses
try:
    for i, wd in enumerate(weight_decay_data["weight_decay"]):
        plt.figure()
        plt.plot(
            weight_decay_data["losses"]["train"][i * 20 : (i + 1) * 20],
            label="Training Loss",
        )
        plt.plot(
            weight_decay_data["losses"]["val"][i * 20 : (i + 1) * 20],
            label="Validation Loss",
        )
        plt.title(f"Losses for Weight Decay={wd} - Synthetic Cosmology")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"losses_wd_{wd}.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss plot for weight decay {wd}: {e}")
    plt.close()

# Plot Bayesian Evidence Ratios
try:
    plt.figure()
    for i, wd in enumerate(weight_decay_data["weight_decay"]):
        ber_values = weight_decay_data["metrics"]["val"][i * 20 : (i + 1) * 20]
        plt.plot(ber_values, label=f"Weight Decay={wd}")
    plt.title("Bayesian Evidence Ratio Across Weight Decays - Synthetic Cosmology")
    plt.xlabel("Epoch")
    plt.ylabel("BER")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "ber_across_weight_decay.png"))
    plt.close()
except Exception as e:
    print(f"Error creating BER plot: {e}")
    plt.close()
