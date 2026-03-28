import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path = os.path.join(
        "experiments/2026-03-27_15-23-06_hubble_tension_transition_attempt_0/logs/0-run/experiment_results/experiment_b6249fc37bf14f77a209bf5b1cf3b368_proc_61158",
        "experiment_data.npy",
    )
    experiment_data = np.load(experiment_data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot mean and standard deviation of training and validation losses
try:
    plt.figure()
    epochs = range(1, 21)
    all_train_losses = []
    all_val_losses = []

    for idx, _ in enumerate(
        experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_cosmology"][
            "hidden_layer_sizes"
        ]
    ):
        train_losses = experiment_data["hyperparam_tuning_hidden_layer_size"][
            "synthetic_cosmology"
        ]["losses"]["train"][idx * 20 : (idx + 1) * 20]
        val_losses = experiment_data["hyperparam_tuning_hidden_layer_size"][
            "synthetic_cosmology"
        ]["losses"]["val"][idx * 20 : (idx + 1) * 20]
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

    mean_train_losses = np.mean(all_train_losses, axis=0)
    std_train_losses = np.std(all_train_losses, axis=0)
    mean_val_losses = np.mean(all_val_losses, axis=0)
    std_val_losses = np.std(all_val_losses, axis=0)

    plt.errorbar(
        epochs,
        mean_train_losses,
        yerr=std_train_losses,
        label="Train Loss (Mean ± Std)",
        capsize=3,
    )
    plt.errorbar(
        epochs,
        mean_val_losses,
        yerr=std_val_losses,
        label="Val Loss (Mean ± Std)",
        capsize=3,
    )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Mean and Std of Training and Validation Losses")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_cosmology_mean_std_loss_curves.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating mean and std loss curves plot: {e}")
    plt.close()

# Plot mean and standard deviation for BER
try:
    plt.figure()
    all_ber_vals = []

    for idx, _ in enumerate(
        experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_cosmology"][
            "hidden_layer_sizes"
        ]
    ):
        ber_vals = experiment_data["hyperparam_tuning_hidden_layer_size"][
            "synthetic_cosmology"
        ]["metrics"]["val"][idx * 20 : (idx + 1) * 20]
        all_ber_vals.append(ber_vals)

    mean_ber_vals = np.mean(all_ber_vals, axis=0)
    std_ber_vals = np.std(all_ber_vals, axis=0)

    plt.errorbar(
        epochs, mean_ber_vals, yerr=std_ber_vals, label="BER (Mean ± Std)", capsize=3
    )
    plt.xlabel("Epochs")
    plt.ylabel("BER")
    plt.title("Mean and Std of Bayesian Evidence Ratio per Epoch")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_cosmology_mean_std_BER_curves.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating mean and std BER plot: {e}")
    plt.close()
