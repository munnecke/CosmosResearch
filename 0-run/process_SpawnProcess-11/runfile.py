import os
import numpy as np


def print_experiment_results():
    working_dir = os.path.join(os.getcwd(), "working")
    experiment_data_file = os.path.join(working_dir, "experiment_data_synthetic.npy")

    # Load the data
    experiment_data_synthetic = np.load(experiment_data_file, allow_pickle=True).item()

    # Extract best or final values for the synthetic cosmology dataset
    # Synthetic Cosmology
    print("Dataset: Synthetic Cosmology")
    # Assuming we take the last epoch value as the final one
    final_validation_ber = experiment_data_synthetic["metrics"]["val"][-1]
    final_train_loss = experiment_data_synthetic["losses"]["train"][-1]
    final_validation_loss = experiment_data_synthetic["losses"]["val"][-1]

    print(f"Final Train Loss: {final_train_loss}")
    print(f"Final Validation Loss: {final_validation_loss}")
    print(f"Final Validation BER: {final_validation_ber}")

    # Additional datasets
    additional_datasets = [
        "KGDataset",
        "Bioasq_B",
    ]  # Should match original dataset names
    for dataset_name in additional_datasets:
        dataset_file = os.path.join(working_dir, f"experiment_data_{dataset_name}.npy")

        try:
            experiment_data = np.load(dataset_file, allow_pickle=True).item()

            # Print dataset name
            print(f"Dataset: {dataset_name}")

            # Assuming final values for demonstration
            final_validation_ber = experiment_data["metrics"]["val"][-1]
            final_train_loss = experiment_data["losses"]["train"][-1]
            final_validation_loss = experiment_data["losses"]["val"][-1]

            print(f"Final Train Loss: {final_train_loss}")
            print(f"Final Validation Loss: {final_validation_loss}")
            print(f"Final Validation BER: {final_validation_ber}")

        except FileNotFoundError:
            print(f"Data for dataset {dataset_name} not found.")


print_experiment_results()
