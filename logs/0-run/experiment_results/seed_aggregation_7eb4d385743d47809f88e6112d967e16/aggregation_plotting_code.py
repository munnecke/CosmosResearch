import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")


def aggregate_and_plot(data, metric, ylabel, title, filename):
    try:
        epochs = range(1, 21)
        mean_values = np.mean(data, axis=0)
        std_errors = np.std(data, axis=0) / np.sqrt(data.shape[0])

        plt.figure()
        plt.errorbar(epochs, mean_values, yerr=std_errors, label="Mean", fmt="-o")
        plt.fill_between(
            epochs,
            mean_values - std_errors,
            mean_values + std_errors,
            alpha=0.2,
            label="Standard Error",
        )
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(working_dir, filename))
        plt.close()

    except Exception as e:
        print(f"Error creating {title} plot: {e}")
        plt.close()


try:
    experiment_data_path_list = [
        "experiments/2026-03-27_15-23-06_hubble_tension_transition_attempt_0/logs/0-run/experiment_results/experiment_1b49dfd86b234d8c968b2fafc6fc3dfb_proc_63283/experiment_data.npy",
        "experiments/2026-03-27_15-23-06_hubble_tension_transition_attempt_0/logs/0-run/experiment_results/experiment_66aa0dbfb38a48c98c7938325ee1300a_proc_63284/experiment_data.npy",
        "experiments/2026-03-27_15-23-06_hubble_tension_transition_attempt_0/logs/0-run/experiment_results/experiment_e48aaa45a77b4f8eb30d6e9118b34d88_proc_63285/experiment_data.npy",
    ]

    all_train_losses = []
    all_val_losses = []
    all_ber_vals = []

    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()

        for idx in range(
            len(
                experiment_data["hyperparam_tuning_hidden_layer_size"][
                    "synthetic_cosmology"
                ]["hidden_layer_sizes"]
            )
        ):
            train_losses = experiment_data["hyperparam_tuning_hidden_layer_size"][
                "synthetic_cosmology"
            ]["losses"]["train"][idx * 20 : (idx + 1) * 20]
            val_losses = experiment_data["hyperparam_tuning_hidden_layer_size"][
                "synthetic_cosmology"
            ]["losses"]["val"][idx * 20 : (idx + 1) * 20]
            ber_vals = experiment_data["hyperparam_tuning_hidden_layer_size"][
                "synthetic_cosmology"
            ]["metrics"]["val"][idx * 20 : (idx + 1) * 20]

            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            all_ber_vals.append(ber_vals)

    all_train_losses = np.array(all_train_losses)
    all_val_losses = np.array(all_val_losses)
    all_ber_vals = np.array(all_ber_vals)

    aggregate_and_plot(
        all_train_losses,
        "losses/train",
        "Loss",
        "Mean Training Loss with Std Error",
        "mean_training_loss.png",
    )
    aggregate_and_plot(
        all_val_losses,
        "losses/val",
        "Loss",
        "Mean Validation Loss with Std Error",
        "mean_validation_loss.png",
    )
    aggregate_and_plot(
        all_ber_vals, "metrics/val", "BER", "Mean BER with Std Error", "mean_BER.png"
    )

except Exception as e:
    print(f"Error loading experiment data: {e}")
