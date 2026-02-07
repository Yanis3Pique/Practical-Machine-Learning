import os
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

folder_curent = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(folder_curent, "Results_Baselines")
os.makedirs(out_dir, exist_ok=True)
feature_types = ["hog", "lbp"]


def knn_train(training_features, training_labels, nr_neighbours, weights, metric):
    KNN = KNeighborsClassifier(
        n_neighbors=nr_neighbours,
        weights=weights,
        metric=metric
    )
    KNN.fit(training_features, training_labels)
    return KNN


def evaluate_accuracy(KNN, testing_features, actual_labels_testing):
    predicted_labels = KNN.predict(testing_features)
    accurayc = accuracy_score(actual_labels_testing, predicted_labels)
    return float(accurayc)


for feature_type in feature_types:
    folder_features = os.path.join(folder_curent, "features_" + feature_type)
    train_features_file = os.path.join(folder_features, "train_" + feature_type + "_features.npy")
    test_features_file = os.path.join(folder_features, "test_" + feature_type + "_features.npy")
    train_labels_file = os.path.join(folder_features, "train_labels.npy")
    test_labels_file = os.path.join(folder_features, "test_labels.npy")
    train_features = np.load(train_features_file).astype(np.float32)
    test_features = np.load(test_features_file).astype(np.float32)
    train_labels = np.load(train_labels_file).astype(np.int64)
    test_labels = np.load(test_labels_file).astype(np.int64)
    print()
    print("KNN")
    print("Feature type:", feature_type)
    print("Folder features:", folder_features)
    print("train_features:", train_features.shape)
    print("test_features:", test_features.shape)
    print("train_labels:", train_labels.shape)
    print("test_labels:", test_labels.shape)

    # we extract some validation data from the original train data so we're not leaking the test one to the model
    training_features, validation_features, training_labels, validation_labels = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    print(training_features.shape, validation_features.shape)

    lista_nr_neighbours = range(1, 500, 2) # hyperparameters
    lista_weights = ["uniform", "distance"]
    lista_metric = ["minkowski", "cosine", "euclidean", "manhattan"]
    rezultate = []

    for nr_neighbours in lista_nr_neighbours:
        for weights in lista_weights:
            for metric in lista_metric:
                KNN = knn_train(training_features, training_labels, nr_neighbours=nr_neighbours, weights=weights, metric=metric)
                accuracy_KNN_on_validation = evaluate_accuracy(KNN, validation_features, validation_labels)

                print("nr_neighbours =", nr_neighbours, "   weights =", weights, "   metric =", metric, "   accuracy_validation =", round(accuracy_KNN_on_validation, 4))
                rezultate.append({
                    "feature_type": feature_type,
                    "nr_neighbours": int(nr_neighbours),
                    "weights": str(weights),
                    "metric": str(metric),
                    "validation_accuracy": float(accuracy_KNN_on_validation),
                })

    dataframe = pd.DataFrame(rezultate)
    dataframe_sorted = dataframe.sort_values(by="validation_accuracy", ascending=False).reset_index(drop=True)

    # validation accuracy vs. nr of neighbors
    plt.figure()
    dataframe["nr_neighbours"] = dataframe["nr_neighbours"].astype(int)
    dataframe["validation_accuracy"] = dataframe["validation_accuracy"].astype(float)

    combinatii_weight_metric = [("uniform", "minkowski"), ("distance", "minkowski"), ("uniform", "cosine"), ("distance", "cosine"),
                                ("uniform", "euclidean"), ("distance", "euclidean"), ("uniform", "manhattan"), ("distance", "manhattan")]
    for weights, metric in combinatii_weight_metric:
        dataframe_for_plotting = dataframe[(dataframe["weights"] == weights) & (dataframe["metric"] == metric)].sort_values("nr_neighbours")
        plt.plot(dataframe_for_plotting["nr_neighbours"].values, dataframe_for_plotting["validation_accuracy"].values, marker="o", label=f"{weights} + {metric}")

    plt.title("KNN validation accuracy vs k - " + str(feature_type))
    plt.xlabel("nr_neighbours (k)")
    plt.ylabel("validation_accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path_figure = os.path.join(out_dir, "knn_curve_" + feature_type + ".png")
    plt.savefig(path_figure)
    plt.close()

    best_row = dataframe_sorted.iloc[0]
    best_nr_neighbours = int(best_row["nr_neighbours"])
    best_weights = best_row["weights"]
    best_metric = best_row["metric"]

    print("Chosen hyperparams by validation set accuracy:")
    print("best_nr_neighbours =", best_nr_neighbours)
    print("best_weights =", best_weights)
    print("best_metric =", best_metric)

    KNN = knn_train(train_features, train_labels, best_nr_neighbours, best_weights, best_metric)
    KNN_test_accuracy = evaluate_accuracy(KNN, test_features, test_labels)

    print("Final KNN:", f"{KNN_test_accuracy}% accuracy")
    dataframe_sorted.to_csv(os.path.join(out_dir, "knn_grid_search_" + feature_type + ".csv"), index=False)