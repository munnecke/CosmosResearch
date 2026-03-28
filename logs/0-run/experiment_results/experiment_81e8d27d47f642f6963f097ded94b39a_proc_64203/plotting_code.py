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

noise_levels = [0.05, 0.1, 0.15, 0.2]
for noise in noise_levels:
    noise_key = f"noise_{noise}"

    try:
        # Plot train and validation losses
        plt.figure()
        epochs = range(
            len(
                experiment_data["data_noise_level_ablation"][noise_key]["losses"][
                    "train"
                ]
            )
        )
        plt.plot(
            epochs,
            experiment_data["data_noise_level_ablation"][noise_key]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            epochs,
            experiment_data["data_noise_level_ablation"][noise_key]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"Loss Curve for Noise Level {noise}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curve_noise_{noise}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for noise {noise}: {e}")
        plt.close()

    try:
        # Plot predictions vs ground truth
        plt.figure()
        predictions = experiment_data["data_noise_level_ablation"][noise_key][
            "predictions"
        ]
        ground_truth = experiment_data["data_noise_level_ablation"][noise_key][
            "ground_truth"
        ]
        plt.scatter(ground_truth, predictions, alpha=0.5)
        plt.title(f"Predictions vs Ground Truth for Noise Level {noise}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(working_dir, f"predictions_vs_ground_truth_noise_{noise}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions vs ground truth plot for noise {noise}: {e}")
        plt.close()
