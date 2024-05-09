import os
import json

# Directory containing experiment results
results_dir = "results"

# Initialize variables to store maximum accuracies and corresponding experiment number
max_train_accuracy = 0
max_test_accuracy = 0
exp_with_max_test_accuracy = None

# Iterate over each experiment directory
for experiment_folder in os.listdir(results_dir):
    if experiment_folder.startswith("exp_"):
        experiment_path = os.path.join(results_dir, experiment_folder)
        json_file_path = os.path.join(experiment_path, "results.json")

        # Check if JSON file exists
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

                # Extract accuracy values
                max_train_acc = max(data["train_accuracies"])
                max_test_acc = max(data["test_accuracies"])

                # Update maximum accuracies and experiment number if necessary
                if max_train_acc > max_train_accuracy:
                    max_train_accuracy = max_train_acc
                if max_test_acc > max_test_accuracy:
                    max_test_accuracy = max_test_acc
                    exp_with_max_test_accuracy = experiment_folder

                print(f"Experiment: {experiment_folder}")
                print(f"Max Train Accuracy: {max_train_acc}")
                print(f"Max Test Accuracy: {max_test_acc}")
                print()

# Print overall maximum test accuracy and corresponding experiment number
print("Overall Maximum Test Accuracy:", max_test_accuracy)
print("Experiment with Maximum Test Accuracy:", exp_with_max_test_accuracy)
