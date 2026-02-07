import os
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score

np.random.seed(42)

root = os.path.dirname(os.path.abspath(__file__))
fcm_folder = os.path.join(root, "Results_Fuzzy_c-means")
os.makedirs(fcm_folder, exist_ok=True)

fuzzy_m = [round(x, 2) for x in np.arange(1.05, 2.51, 0.05)]
distance_metrics = ["euclidean", "cityblock", "cosine"]

def fuzzy_cmeans_train_and_predict(train_features, test_features, fuzziness_m, distance_metric):
    train_centroids, train_membership_matrix, _, _, _, _, train_fpc = fuzz.cluster.cmeans(
        train_features.T, c=2, m=fuzziness_m, metric=distance_metric, error=1e-5, maxiter=500
    )
    test_membership_matrix, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        test_features.T, train_centroids, m=fuzziness_m, metric=distance_metric, error=1e-5, maxiter=500
    )
    return train_centroids, train_membership_matrix, train_fpc, test_membership_matrix

def majority_vote_mapping(train_clusters, train_labels):
    counts = np.zeros((2, 2)) # we try both mapping configurations on the training features and we pick the best one
    for cluster_id in range(2):
        indexes = np.where(train_clusters == cluster_id)[0]
        labels = train_labels[indexes]
        for label in range(2):
            counts[cluster_id, label] = np.sum(labels == label)
    if (counts[0, 1] + counts[1, 0]) > (counts[0, 0] + counts[1, 1]):
        return {0: 1, 1: 0}
    else:
        return {0: 0, 1: 1}

def main():
    for feature_type in ["hog", "lbp"]:
        train_features = np.load(os.path.join(os.path.join(root, f"features_{feature_type}"), f"train_{feature_type}_features.npy"))
        test_features = np.load(os.path.join(os.path.join(root, f"features_{feature_type}"), f"test_{feature_type}_features.npy"))
        train_labels = np.load(os.path.join(os.path.join(root, f"features_{feature_type}"), "train_labels.npy"))
        test_labels = np.load(os.path.join(os.path.join(root, f"features_{feature_type}"), "test_labels.npy"))
        print(f"\nFCM - {feature_type}")
        print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)

        all_tries = []
        for metric in distance_metrics: # for every combination of fuzzy parameter m and metric type we do
            for m in fuzzy_m:
                train_centroids, train_membership_matrix, train_fpc, test_membership_matrix = fuzzy_cmeans_train_and_predict(
                    train_features, test_features, m, metric
                )

                train_clusters = np.argmax(train_membership_matrix, axis=0)
                test_clusters = np.argmax(test_membership_matrix, axis=0)

                mapping = majority_vote_mapping(train_clusters, train_labels)

                train_predicted_labels = np.vectorize(mapping.get)(train_clusters)
                test_predicted_labels = np.vectorize(mapping.get)(test_clusters)

                dataframe_result = {
                    "nr_clustere": 2,
                    "distance_metric": metric,
                    "fuzziness_m": m,
                    "fpc_train": train_fpc,
                    "mapping": mapping,
                    "train_accuracy": accuracy_score(train_labels, train_predicted_labels),
                    "test_accuracy": accuracy_score(test_labels, test_predicted_labels),
                    "train_ari": adjusted_rand_score(train_labels, train_clusters),
                    "test_ari": adjusted_rand_score(test_labels, test_clusters),
                    "train_nmi": normalized_mutual_info_score(train_labels, train_clusters),
                    "test_nmi": normalized_mutual_info_score(test_labels, test_clusters)
                }
                all_tries.append(dataframe_result)

        all_tries = pd.DataFrame(all_tries)
        all_tries.to_csv(os.path.join(fcm_folder, f"all_tries_{feature_type}.csv"), index=False)

        best_index = all_tries["fpc_train"].idxmax()
        best_metric = all_tries.loc[best_index, "distance_metric"]
        best_m = all_tries.loc[best_index, "fuzziness_m"]
        best_fpc = all_tries.loc[best_index, "fpc_train"]

        print("best:", best_metric, "m =", best_m, ", FPC(train) =", best_fpc)

        # we save FCP curve
        plt.figure()
        for metric in distance_metrics:
            sub = all_tries[all_tries["distance_metric"] == metric].sort_values("fuzziness_m")
            plt.plot(sub["fuzziness_m"].values, sub["fpc_train"].values, label=metric)
        plt.axvline(best_m, linestyle="--")
        plt.title(f"FPC vs m - train - {feature_type}")
        plt.xlabel("fuzziness_m")
        plt.ylabel("fpc_train")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fcm_folder, f"fpc_curve_{feature_type}.png"))
        plt.close()

        train_centroids, train_membership_matrix, train_fpc, test_membership_matrix = fuzzy_cmeans_train_and_predict(
            train_features, test_features, best_m, best_metric
        )

        # we save the look of the train cluster, by converting it to 2D with PCA
        pca = PCA(2, random_state=42)
        train_features_pca = pca.fit_transform(train_features)
        train_centroids_pca = pca.transform(train_centroids)
        train_clusters = np.argmax(train_membership_matrix, axis=0)

        plt.figure()
        plt.scatter(train_features_pca[:, 0], train_features_pca[:, 1], c=train_clusters, s=10)
        plt.scatter(train_centroids_pca[:, 0], train_centroids_pca[:, 1], marker="x", s=120)
        plt.title(f"PCA train clusters - {feature_type}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(fcm_folder, f"pca_clusters_{feature_type}.png"))
        plt.close()

if __name__ == "__main__":
    main()