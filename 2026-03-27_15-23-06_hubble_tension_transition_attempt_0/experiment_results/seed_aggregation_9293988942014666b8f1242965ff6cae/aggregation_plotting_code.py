import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    # Load data from multiple experiment runs to compute aggregate metrics
    experiment_data_path_list = [
        "experiments/2026-03-27_15-23-06_hubble_tension_transition_attempt_0/logs/0-run/experiment_results/experiment_2156ca031d0741aa9b3cf02761ab1f58_proc_41165/experiment_data.npy",
        "experiments/2026-03-27_15-23-06_hubble_tension_transition_attempt_0/logs/0-run/experiment_results/experiment_c2b96f9f7e05403283ad390c0798aa91_proc_41158/experiment_data.npy",
    ]

    all_train_losses = []
    all_val_losses = []
    all_metrics = []

    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(experiment_data_path, allow_pickle=True).item()
        all_train_losses.append(
            experiment_data["synthetic_cosmology"]["losses"]["train"]
        )
        all_val_losses.append(experiment_data["synthetic_cosmology"]["losses"]["val"])
        all_metrics.append(experiment_data["synthetic_cosmology"]["metrics"]["val"])

    # Convert data to arrays for aggregation
    all_train_losses = np.array(all_train_losses)
    all_val_losses = np.array(all_val_losses)
    all_metrics = np.array(all_metrics)

    epochs = range(1, all_train_losses.shape[1] + 1)

except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot aggregated training and validation losses with standard error
try:
    plt.figure()
    mean_train_loss = np.mean(all_train_losses, axis=0)
    mean_val_loss = np.mean(all_val_losses, axis=0)
    std_train_loss = np.std(all_train_losses, axis=0) / np.sqrt(
        all_train_losses.shape[0]
    )
    std_val_loss = np.std(all_val_losses, axis=0) / np.sqrt(all_val_losses.shape[0])

    plt.plot(epochs, mean_train_loss, label="Mean Training Loss")
    plt.fill_between(
        epochs,
        mean_train_loss - std_train_loss,
        mean_train_loss + std_train_loss,
        alpha=0.2,
    )
    plt.plot(epochs, mean_val_loss, label="Mean Validation Loss")
    plt.fill_between(
        epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.2
    )
    plt.title("Aggregated Loss Curves with Standard Error")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "aggregated_synthetic_cosmology_loss_curves.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves plot: {e}")
    plt.close()

# Plot aggregated BER metric with standard error
try:
    plt.figure()
    mean_metric = np.mean(all_metrics, axis=0)
    std_metric = np.std(all_metrics, axis=0) / np.sqrt(all_metrics.shape[0])

    plt.plot(epochs, mean_metric, label="Mean Validation BER")
    plt.fill_between(
        epochs, mean_metric - std_metric, mean_metric + std_metric, alpha=0.2
    )
    plt.title("Aggregated BER over Epochs with Standard Error")
    plt.xlabel("Epochs")
    plt.ylabel("BER")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "aggregated_synthetic_cosmology_ber.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated BER plot: {e}")
    plt.close()
