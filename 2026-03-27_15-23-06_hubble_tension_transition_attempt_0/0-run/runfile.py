working_dir


import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

experiment_data_path_list = [
    "experiments/2026-03-27_15-23-06_hubble_tension_transition_attempt_0/logs/0-run/experiment_results/experiment_ed634cf32ac14a8eb3e27e9d8bab38cc_proc_64229/experiment_data.npy",
    "experiments/2026-03-27_15-23-06_hubble_tension_transition_attempt_0/logs/0-run/experiment_results/experiment_f2e2c90c869f4a8ba31bc6334e718366_proc_64176/experiment_data.npy",
    # Add more paths if available
]

experiment_data_list = []

try:
    for path in experiment_data_path_list:
        experiment_data = np.load(path, allow_pickle=True).item()
        experiment_data_list.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Plot mean and standard error of training and validation losses
    plt.figure()
    epochs = range(1, 21)
    train_loss_aggregated = []
    val_loss_aggregated = []

    for i in range(len(epochs)):
        losses_epoch_train = [
            exp_data["hyperparam_tuning_hidden_layer_size"]["synthetic_cosmology"][
                "losses"
            ]["train"][i]
            for exp_data in experiment_data_list
        ]
        losses_epoch_val = [
            exp_data["hyperparam_tuning_hidden_layer_size"]["synthetic_cosmology"][
                "losses"
            ]["val"][i]
            for exp_data in experiment_data_list
        ]
        train_loss_aggregated.append(
            (
                np.mean(losses_epoch_train),
                np.std(losses_epoch_train) / np.sqrt(len(losses_epoch_train)),
            )
        )
        val_loss_aggregated.append(
            (
                np.mean(losses_epoch_val),
                np.std(losses_epoch_val) / np.sqrt(len(losses_epoch_val)),
            )
        )

    mean_train_loss, std_err_train_loss = zip(*train_loss_aggregated)
    mean_val_loss, std_err_val_loss = zip(*val_loss_aggregated)

    plt.errorbar(
        epochs,
        mean_train_loss,
        yerr=std_err_train_loss,
        label="Mean Train Loss",
        fmt="-o",
    )
    plt.errorbar(
        epochs, mean_val_loss, yerr=std_err_val_loss, label="Mean Val Loss", fmt="-o"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Mean Training and Validation Loss Curves with Standard Error")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mean_synthetic_cosmology_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean loss curves plot: {e}")
    plt.close()

try:
    # Plot mean and standard error of BER values
    plt.figure()
    ber_aggregated = []

    for i in range(len(epochs)):
        ber_vals_epoch = [
            exp_data["hyperparam_tuning_hidden_layer_size"]["synthetic_cosmology"][
                "metrics"
            ]["val"][i]
            for exp_data in experiment_data_list
        ]

        ber_aggregated.append(
            (
                np.mean(ber_vals_epoch),
                np.std(ber_vals_epoch) / np.sqrt(len(ber_vals_epoch)),
            )
        )

    mean_ber, std_err_ber = zip(*ber_aggregated)

    plt.errorbar(epochs, mean_ber, yerr=std_err_ber, label="Mean BER", fmt="-o")

    plt.xlabel("Epochs")
    plt.ylabel("BER")
    plt.title("Mean Bayesian Evidence Ratio per Epoch with Standard Error")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mean_synthetic_cosmology_BER_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean BER plot: {e}")
    plt.close()

try:
    # Plot predictions vs. ground truth with aggregation if needed
    plt.figure()
    all_predictions = []
    all_ground_truth = []

    for exp_data in experiment_data_list:
        predictions = exp_data["hyperparam_tuning_hidden_layer_size"][
            "synthetic_cosmology"
        ]["predictions"]
        ground_truth = exp_data["hyperparam_tuning_hidden_layer_size"][
            "synthetic_cosmology"
        ]["ground_truth"]
        all_predictions.extend(predictions)
        all_ground_truth.extend(ground_truth)

    plt.scatter(all_ground_truth, all_predictions, alpha=0.2)
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Aggregated Predictions vs. Ground Truth")
    plt.savefig(
        os.path.join(
            working_dir, "aggregate_synthetic_cosmology_predictions_vs_ground_truth.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
