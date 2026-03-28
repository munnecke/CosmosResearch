import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Access the relevant data
losses_train = experiment_data["ablation_study_model_depth"]["synthetic_cosmology"][
    "losses"
]["train"]
losses_val = experiment_data["ablation_study_model_depth"]["synthetic_cosmology"][
    "losses"
]["val"]
predictions = experiment_data["ablation_study_model_depth"]["synthetic_cosmology"][
    "predictions"
]
ground_truth = experiment_data["ablation_study_model_depth"]["synthetic_cosmology"][
    "ground_truth"
]
hidden_layers = experiment_data["ablation_study_model_depth"]["synthetic_cosmology"][
    "hidden_layers"
]

# Plot training and validation losses
try:
    epochs = np.arange(len(losses_train))
    plt.figure()
    plt.plot(epochs, losses_train, label="Training Loss")
    plt.plot(epochs, losses_val, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_synthetic_cosmology.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot predictions vs. ground truth at different depths (simple scatter plot for simplicity)
try:
    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Predictions vs. Ground Truth")
    plt.savefig(
        os.path.join(working_dir, "predictions_vs_ground_truth_synthetic_cosmology.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs. ground truth plot: {e}")
    plt.close()
