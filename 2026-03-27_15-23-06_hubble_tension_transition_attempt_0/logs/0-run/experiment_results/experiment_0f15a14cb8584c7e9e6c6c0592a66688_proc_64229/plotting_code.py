import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load the experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses
try:
    plt.figure()
    epochs = range(
        1,
        len(
            experiment_data["batch_size_variation"]["synthetic_cosmology"]["losses"][
                "train"
            ]
        )
        // len(
            experiment_data["batch_size_variation"]["synthetic_cosmology"][
                "batch_sizes"
            ]
        )
        + 1,
    )
    for i, batch_size in enumerate(
        experiment_data["batch_size_variation"]["synthetic_cosmology"]["batch_sizes"]
    ):
        start = i * len(epochs)
        plt.plot(
            epochs,
            experiment_data["batch_size_variation"]["synthetic_cosmology"]["losses"][
                "train"
            ][start : start + len(epochs)],
            label=f"Train Loss, Batch Size={batch_size}",
        )
        plt.plot(
            epochs,
            experiment_data["batch_size_variation"]["synthetic_cosmology"]["losses"][
                "val"
            ][start : start + len(epochs)],
            label=f"Val Loss, Batch Size={batch_size}",
            linestyle="--",
        )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_cosmology_loss_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot Bayesian Evidence Ratio (BER)
try:
    plt.figure()
    for i, batch_size in enumerate(
        experiment_data["batch_size_variation"]["synthetic_cosmology"]["batch_sizes"]
    ):
        start = i * len(epochs)
        plt.plot(
            epochs,
            experiment_data["batch_size_variation"]["synthetic_cosmology"]["metrics"][
                "val"
            ][start : start + len(epochs)],
            label=f"BER, Batch Size={batch_size}",
        )
    plt.title("Bayesian Evidence Ratio (BER) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("BER")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_cosmology_BER_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating BER plot: {e}")
    plt.close()

# Plot ground truth vs predictions for the last epoch of each batch size setup
try:
    plt.figure()
    for i, batch_size in enumerate(
        experiment_data["batch_size_variation"]["synthetic_cosmology"]["batch_sizes"]
    ):
        start = i * int(
            len(
                experiment_data["batch_size_variation"]["synthetic_cosmology"][
                    "predictions"
                ]
            )
            / len(
                experiment_data["batch_size_variation"]["synthetic_cosmology"][
                    "batch_sizes"
                ]
            )
        )
        end = (i + 1) * int(
            len(
                experiment_data["batch_size_variation"]["synthetic_cosmology"][
                    "predictions"
                ]
            )
            / len(
                experiment_data["batch_size_variation"]["synthetic_cosmology"][
                    "batch_sizes"
                ]
            )
        )
        plt.scatter(
            experiment_data["batch_size_variation"]["synthetic_cosmology"][
                "ground_truth"
            ][start:end],
            experiment_data["batch_size_variation"]["synthetic_cosmology"][
                "predictions"
            ][start:end],
            label=f"Batch Size={batch_size}",
            alpha=0.5,
        )
    plt.title("Ground Truth vs Predictions")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_cosmology_ground_truth_vs_predictions.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()
