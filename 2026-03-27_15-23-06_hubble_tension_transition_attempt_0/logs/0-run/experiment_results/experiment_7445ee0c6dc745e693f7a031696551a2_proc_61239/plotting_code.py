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

for optimizer in experiment_data.keys():

    # Plot training and validation loss curves
    try:
        plt.figure()
        train_losses = experiment_data[optimizer]["synthetic_cosmology"]["losses"][
            "train"
        ]
        val_losses = experiment_data[optimizer]["synthetic_cosmology"]["losses"]["val"]
        epochs = np.arange(len(train_losses))
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"{optimizer} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{optimizer}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for {optimizer}: {e}")
        plt.close()

    # Plot Bayesian Evidence Ratio (BER)
    try:
        plt.figure()
        ber_values = experiment_data[optimizer]["synthetic_cosmology"]["metrics"]["val"]
        plt.plot(epochs, ber_values, label="BER")
        plt.title(f"{optimizer} BER over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Bayesian Evidence Ratio")
        plt.savefig(os.path.join(working_dir, f"{optimizer}_BER.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating BER plot for {optimizer}: {e}")
        plt.close()

    # Plot predictions vs ground truth at intervals
    try:
        predictions = np.array(
            experiment_data[optimizer]["synthetic_cosmology"]["predictions"]
        )
        ground_truth = np.array(
            experiment_data[optimizer]["synthetic_cosmology"]["ground_truth"]
        )
        for epoch in range(0, len(train_losses), 5):
            plt.figure()
            plt.scatter(
                ground_truth[
                    epoch
                    * len(ground_truth)
                    // len(train_losses) : (epoch + 1)
                    * len(ground_truth)
                    // len(train_losses)
                ],
                predictions[
                    epoch
                    * len(predictions)
                    // len(train_losses) : (epoch + 1)
                    * len(predictions)
                    // len(train_losses)
                ],
                alpha=0.5,
                label="Data Points",
            )
            plt.plot(
                [ground_truth.min(), ground_truth.max()],
                [ground_truth.min(), ground_truth.max()],
                "k--",
                label="Ideal",
            )
            plt.title(f"{optimizer} Predictions vs Ground Truth at Epoch {epoch}")
            plt.xlabel("Ground Truth")
            plt.ylabel("Predictions")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{optimizer}_pred_vs_gt_epoch_{epoch}.png")
            )
            plt.close()
    except Exception as e:
        print(f"Error creating prediction vs ground truth plot for {optimizer}: {e}")
        plt.close()
