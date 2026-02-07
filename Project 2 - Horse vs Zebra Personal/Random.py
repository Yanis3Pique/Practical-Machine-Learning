import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

random.seed(42)
np.random.seed(42)

folder_curent = os.path.dirname(os.path.abspath(__file__))
folder_dataset = os.path.join(folder_curent, "Horse2zebra")
metadatacsv = os.path.join(folder_dataset, "metadata.csv")

out_dir = os.path.join(folder_curent, "Results_Baselines")
os.makedirs(out_dir, exist_ok=True)

metadata = pd.read_csv(metadatacsv)

labeling_domeniu = {"A (Horse)": 0, "B (Zebra)": 1} # A = horse -> 0, B = zebra -> 1
metadata["label"] = metadata["domain"].map(labeling_domeniu)

train_labels = metadata[metadata["split"] == "train"]["label"].astype(np.int64).to_numpy()
test_labels = metadata[metadata["split"] == "test"]["label"].astype(np.int64).to_numpy()

labelsH0Z1 = np.unique(train_labels).astype(np.int64) # [0, 1]
sansa = 1.0 / float(len(labelsH0Z1))

print("Random baseline")
print("Apriori chance =", sansa)
print("Train size:", len(train_labels))
print("Test size:", len(test_labels))

nr_trials = 1000
rezultate = []

for t in range(nr_trials):
    predicted_labels_training = np.random.choice(labelsH0Z1, size=len(train_labels), replace=True).astype(np.int64)
    predicted_labels_testing = np.random.choice(labelsH0Z1, size=len(test_labels), replace=True).astype(np.int64)

    acuratete_train = accuracy_score(train_labels, predicted_labels_training)
    acuratete_test = accuracy_score(test_labels, predicted_labels_testing)

    rezultate.append({
        "trial_nr": int(t),
        "apriori_chance": float(sansa),
        "train_accuracy_score": float(acuratete_train),
        "test_accuracy_score": float(acuratete_test),
    })

df = pd.DataFrame(rezultate)

summary = pd.DataFrame([{
    "model": "random_uniform",
    "nr_trials": int(nr_trials),
    "apriori_chance": float(sansa),
    "train_accuracy_mean": float(df["train_accuracy_score"].mean()),
    "train_accuracy_std": float(df["train_accuracy_score"].std()),
    "test_accuracy_mean": float(df["test_accuracy_score"].mean()),
    "test_accuracy_std": float(df["test_accuracy_score"].std()),
}])

trials_csv_file = os.path.join(out_dir, "random_baseline_trials.csv")
summary_csv_file = os.path.join(out_dir, "random_baseline_summary.csv")

df.to_csv(trials_csv_file, index=False)
summary.to_csv(summary_csv_file, index=False)

print()
print("Saved:", trials_csv_file)
print("Saved:", summary_csv_file)
print()
print("Summary:")
print(summary.to_string(index=False))