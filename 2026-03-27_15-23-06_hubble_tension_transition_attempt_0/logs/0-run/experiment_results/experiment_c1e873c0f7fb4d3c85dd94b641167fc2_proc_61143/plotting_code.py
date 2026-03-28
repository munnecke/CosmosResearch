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

for lr, data in experiment_data.items():
    try:
        plt.figure()
        plt.plot(data["synthetic_cosmology"]["losses"]["train"], label="Training Loss")
        plt.plot(data["synthetic_cosmology"]["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss for LR={lr}")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"losses_lr_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for LR {lr}: {e}")
        plt.close()

    predictions = data["synthetic_cosmology"]["predictions"]
    ground_truth = data["synthetic_cosmology"]["ground_truth"]

    intervals = len(predictions) // 5
    for i in range(0, len(predictions), intervals):
        try:
            plt.figure()
            plt.scatter(
                ground_truth[i : i + intervals],
                predictions[i : i + intervals],
                alpha=0.5,
            )
            plt.xlabel("Ground Truth")
            plt.ylabel("Predictions")
            plt.title(f"Predictions vs Ground Truth for LR={lr} (Sampled)")
            plt.savefig(
                os.path.join(
                    working_dir,
                    f"predictions_ground_truth_lr_{lr}_epoch_{i//intervals}.png",
                )
            )
            plt.close()
        except Exception as e:
            print(
                f"Error creating predictions vs ground truth plot for LR {lr} at interval {i}: {e}"
            )
            plt.close()
