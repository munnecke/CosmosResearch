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
    for init_strategy in experiment_data["weight_initialization_ablation"][
        "synthetic_cosmology"
    ]["initialization_strategies"]:
        index = experiment_data["weight_initialization_ablation"][
            "synthetic_cosmology"
        ]["initialization_strategies"].index(init_strategy)

        # Plot training and validation loss curves
        plt.figure()
        plt.plot(
            experiment_data["weight_initialization_ablation"]["synthetic_cosmology"][
                "losses"
            ]["train"][index * 20 : (index + 1) * 20],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["weight_initialization_ablation"]["synthetic_cosmology"][
                "losses"
            ]["val"][index * 20 : (index + 1) * 20],
            label="Validation Loss",
        )
        plt.title(f"Losses for Initialization Strategy: {init_strategy}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curve_{init_strategy}.png"))
        plt.close()
except Exception as e:
    print(f"Error creating plots for initialization strategies: {e}")
    plt.close()
