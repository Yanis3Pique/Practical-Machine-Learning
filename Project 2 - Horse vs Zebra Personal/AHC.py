import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

root = os.path.dirname(os.path.abspath(__file__))
ahc_folder = os.path.join(root, "Results_Agglomerative")
os.makedirs(ahc_folder, exist_ok=True)

hyperparameter_combinations = [("ward", "euclidean"), ("average", "cosine"), ("single", "cosine"), ("complete", "cosine")]

def compute_cluster_centroids(train_features, train_clusters):
    # we need to compute the centroid for each cluster so we can use it later to predict test cluster membership
    centroids = []
    for cluster_id in [0, 1]:
        indexes = np.where(train_clusters == cluster_id)[0]
        centroids.append(train_features[indexes].mean(axis=0))
    return np.vstack(centroids)


def predict_clusters_by_closest_centroid(test_features, centroids, distance_metric):
    # we find the closest centroid, because Agglomerative has no predict() by default
    # distance - shape (N_test, 2)
    distance = cdist(test_features, centroids, metric=distance_metric)
    test_clusters = np.argmin(distance, axis=1)
    return test_clusters

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
        print(f"\nAHC - {feature_type}")
        print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)

        all_tries = []
        for linkage, metric in hyperparameter_combinations: # for every combination of linkage and metric
            AHC_model = AgglomerativeClustering(n_clusters=2, linkage=linkage, metric=metric)

            train_clusters = AHC_model.fit_predict(train_features)
            centroids = compute_cluster_centroids(train_features, train_clusters)
            test_clusters = predict_clusters_by_closest_centroid(test_features, centroids, metric)

            mapping = majority_vote_mapping(train_clusters, train_labels)

            train_predicted_labels = np.vectorize(mapping.get)(train_clusters)
            test_predicted_labels = np.vectorize(mapping.get)(test_clusters)

            silhouette = silhouette_score(train_features, train_clusters, metric=metric)

            dataframe_result = {
                "nr_clustere": 2,
                "linkage_method": linkage,
                "distance_metric": metric,
                "silhouette_train": silhouette,
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
        all_tries.to_csv(os.path.join(ahc_folder, f"all_tries_{feature_type}.csv"), index=False)

        best_index = all_tries["silhouette_train"].idxmax()
        best_linkage = all_tries.loc[best_index, "linkage_method"]
        best_metric = all_tries.loc[best_index, "distance_metric"]
        best_silhouette = all_tries.loc[best_index, "silhouette_train"]

        print("best:", best_linkage, best_metric, ", silhouette =", best_silhouette)

        # we save train silhouette curve
        plt.figure()
        plt.plot(np.arange(len(all_tries)), all_tries["silhouette_train"].values)
        plt.xticks(np.arange(len(all_tries)), all_tries["linkage_method"] + " + " + all_tries["distance_metric"].values, rotation=45)
        plt.axvline(all_tries.index.get_loc(best_index), linestyle="--")
        plt.title(f"Train silhouette - {feature_type}")
        plt.xlabel("configuration")
        plt.ylabel("silhouette_train")
        plt.tight_layout()
        plt.savefig(os.path.join(ahc_folder, f"silhouette_curve_{feature_type}.png"))
        plt.close()

        AHC_best = AgglomerativeClustering(2, linkage=best_linkage, metric=best_metric)
        train_clusters = AHC_best.fit_predict(train_features)
        train_centroids = compute_cluster_centroids(train_features, train_clusters)

        # we save the look of the train cluster, by converting it to 2D with PCA
        pca = PCA(2, random_state=42)
        train_features_pca = pca.fit_transform(train_features)
        train_centroids_pca = pca.transform(train_centroids)

        plt.figure()
        plt.scatter(train_features_pca[:, 0], train_features_pca[:, 1], c=train_clusters, s=10)
        plt.scatter(train_centroids_pca[:, 0], train_centroids_pca[:, 1], marker="x", s=120)
        plt.title(f"PCA train clusters - {feature_type}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(ahc_folder, f"pca_clusters_{feature_type}.png"))
        plt.close()

if __name__ == "__main__":
    main()