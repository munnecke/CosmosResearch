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
    # Plot training and validation losses
    plt.figure()
    for act in experiment_data["activation_ablation"]["synthetic_cosmology"][
        "activations"
    ]:
        train_losses = experiment_data["activation_ablation"]["synthetic_cosmology"][
            "losses"
        ]["train"]
        val_losses = experiment_data["activation_ablation"]["synthetic_cosmology"][
            "losses"
        ]["val"]
        plt.plot(train_losses, label=f"{act} Train")
        plt.plot(val_losses, label=f"{act} Val")
    plt.title("Training and Validation Losses per Activation Function")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_cosmology_loss_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Plot Bayesian Evidence Ratio
    plt.figure()
    for act in experiment_data["activation_ablation"]["synthetic_cosmology"][
        "activations"
    ]:
        BER = experiment_data["activation_ablation"]["synthetic_cosmology"]["metrics"][
            "val"
        ]
        plt.plot(BER, label=f"{act} BER")
    plt.title("Bayesian Evidence Ratio per Activation Function")
    plt.xlabel("Epoch")
    plt.ylabel("BER")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_cosmology_ber_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating BER plot: {e}")
    plt.close()
