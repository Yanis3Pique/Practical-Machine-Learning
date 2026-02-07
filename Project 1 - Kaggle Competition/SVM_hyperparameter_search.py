import os
import random
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau
from joblib import Parallel, delayed

seed = 42 # fixed seed so I can safely reproduce and compare results of the experiments
random.seed(seed)
np.random.seed(seed)

HOME_FOLDER = os.path.dirname(os.path.abspath(__file__)) # absolute path of the python project
TRAIN_CSV_FILE = os.path.join(HOME_FOLDER, "train.csv") # train csv from Kaggle
VALIDATION_CSV_FILE = os.path.join(HOME_FOLDER, "validation.csv") # validation csv from Kaggle
TEST_CSV_FILE = os.path.join(HOME_FOLDER, "test.csv") # test csv from Kaggle
SAMPLES_FOLDER = os.path.join(HOME_FOLDER, "samples") # images folder from Kaggle

train_dataset = pd.read_csv(TRAIN_CSV_FILE) # I load the datasets from the csvs
validation_dataset = pd.read_csv(VALIDATION_CSV_FILE)
test_dataset = pd.read_csv(TEST_CSV_FILE)

print(train_dataset.shape, validation_dataset.shape, test_dataset.shape)

def load_image(id_imagine1): # this function loads one image as a numpy array based on the id of that image
    path = os.path.join(SAMPLES_FOLDER, id_imagine1 + ".npy") # full path of the image
    np_image_array = np.load(path).astype(np.float32) # transforming it into the numpy array + float32 precision
    return np_image_array

toate_ids = ( # all image ids appearing in any split, using set so there's no duplicates
    set(train_dataset["id_noise_1"]) |
    set(train_dataset["id_noise_2"]) |
    set(validation_dataset["id_noise_1"]) |
    set(validation_dataset["id_noise_2"]) |
    set(test_dataset["id_noise_1"]) |
    set(test_dataset["id_noise_2"])
)

training_ids = ( # same, but only for training here
    set(train_dataset["id_noise_1"]) |
    set(train_dataset["id_noise_2"])
)

# keeping a dictionary of all the image samples in memory for easier access
dictionar_imagini = {} # (longer to compute, but given the number of images, it should be fine)
for id_imagine in toate_ids:
    dictionar_imagini[id_imagine] = load_image(id_imagine)

# same as before, only for training images this time
dictionar_training_imagini = {}
for id_imagine in training_ids:
    dictionar_training_imagini[id_imagine] = dictionar_imagini[id_imagine]

print("Number of images:", len(dictionar_imagini))

# print(len(dictionar_imagini))

# sum of pixels for all the images, sum of squared pixels and all number of pixels
# will be used later for calculating mean and standard deviation
suma_pixeli = 0.0
suma_pixeli_squared = 0.0
nr_pixeli = 0

for image in dictionar_training_imagini.values(): # images of size 256 x 256 -> only from the training set
    image_pixels = image.astype(np.float32) # pixel array
    suma_pixeli += float(image_pixels.sum()) # sum of all pixels
    suma_pixeli_squared += float((image_pixels ** 2).sum()) # sum of all pixels squared
    nr_pixeli += image_pixels.size # total number of pixels

# mean, variance and standard deviation of the training set of images
# will be used later in normalizing the training set
mean = suma_pixeli / nr_pixeli
variance = (suma_pixeli_squared / nr_pixeli) - (mean ** 2)
std = float(np.sqrt(max(variance, 1e-8)))

def block_manual_pooling(image_2d, pool_size=16, type="mean"): # split the 2d image in pool_size x pool_size chunks and do mean on those
    height, width = image_2d.shape # height and width of the image = 256
    height_block_nr = height // pool_size # 256/16 = 16 -> basically the block height of the future pooling
    width_block_nr = width // pool_size # 256/16 = 16 -> basically the block width of the future pooling

    # 256 x 256 image with pool_size = 16 becomes
    # (16 blocks tall, each block 16 pixels) x (16 blocks wide, each block 16 pixels)
    reshaped_image = image_2d.reshape(height_block_nr, pool_size, width_block_nr, pool_size) # shape (16, 16, 16, 16)

    if type == "mean":
        pooled = reshaped_image.mean(axis=(1, 3)) # (16, 16) = average of 16 x 16 blocks

    return pooled # (16, 16)

def extract_image_pair_features(id1, id2):
    im1 = dictionar_imagini[id1].astype(np.float32) # loading the two images from the dictionary given their ids
    im2 = dictionar_imagini[id2].astype(np.float32) # forcing them to be float32

    im1 = (im1 - mean) / (std + 1e-8) # normalized images with mean and std from train images only
    im2 = (im2 - mean) / (std + 1e-8)

    # diference and product between images pixel by pixel
    d = im1 - im2 # 256x256 -> difference = how much the images differ
    p = im1 * im2 # 256x256 -> product = how much the images agree in intensity

    # downsampling - mean pooling over 16 pixels(256 / 16 = 16) -> we can't just feed 256^2 values into the model
    d_pooled = block_manual_pooling(d, 16, "mean") # 16 x 16 = 256
    p_pooled = block_manual_pooling(p, 16, "mean") # 16 x 16

    d_downsampled = d_pooled.reshape(-1) # 256 shaped vector from matrix
    p_downsampled = p_pooled.reshape(-1) # 256

    mean1 = im1.mean() # mean and standard deviation stats for the whole, both images
    mean2 = im2.mean()
    std1 = im1.std()
    std2 = im2.std()

    absolute_difference = np.abs(d) # distance statistics based on difference between images
    mean_absolute_difference = absolute_difference.mean()
    max_absolute_difference = absolute_difference.max()
    l1_distance = absolute_difference.sum() # Manhattan distance
    l2_distance = np.sqrt((d ** 2).sum()) # Euclidean distance

    # cosine similarity between images = treat images as vectors and measure the angle between them
    flat_im1 = im1.reshape(-1)
    flat_im2 = im2.reshape(-1)
    p_im1_im2 = np.dot(flat_im1, flat_im2) # the product tells us how aligned the two images vectors are
    p_normim1_normim2 = (np.linalg.norm(flat_im1) * np.linalg.norm(flat_im2) + 1e-8) # normalizes dot product of numerator
    cosine_similarity = p_im1_im2 / p_normim1_normim2

    # feature vector (521 dimensions)
    features = np.concatenate([
        d_downsampled, # 256 pooled difference features
        p_downsampled, # 256 pooled product features
        np.array([mean1, mean2, std1, std2, mean_absolute_difference, max_absolute_difference,
                         l1_distance, l2_distance, cosine_similarity], dtype=np.float32) # 9 scalar stats
    ]) # 256 + 256 + 9 = 521 features/dimensions, in the end

    return features

def feature_matrix_creator(df):
    feature_vectors = [] # storing the feature vectors per image pair
    labels = [] # storing the ground-truth labels

    for i, row in df.iterrows(): # go row by row through the dataframe(train or validation data, depends on the use case)
        id1 = row["id_noise_1"] # the two ids of the images in the image pair
        id2 = row["id_noise_2"]
        label = int(row["label"]) # and their 0/1 true label

        features_vector = extract_image_pair_features(id1, id2) # extract the 521 dimensions feature vector
        feature_vectors.append(features_vector) # store it in the feature vector of vectors
        labels.append(label) # and store the true labels

    matrix_of_features = np.vstack(feature_vectors).astype(np.float32) # transform the vector of vectors into a numpy matrix
    array_of_labels = np.array(labels, dtype=np.int64) # store labels array as a numpy array
    return matrix_of_features, array_of_labels

def train_evaluate_SVM(train_features_std, train_labels, val_features_std, val_labels, C, gamma, use_proba=False): # Results-SVM cu kernel RBF
    #train the SVM with the chosen configuration of hyperparameters
    modelSVM = SVC(kernel="rbf", C=C, gamma=gamma, probability=use_proba, random_state=seed, cache_size=2000)
    modelSVM.fit(train_features_std, train_labels)

    # check accuracy for the trained data
    train_pred = modelSVM.predict(train_features_std)
    accuracy_train = accuracy_score(train_labels, train_pred)

    # check accuracy for validation data now -> check generalization capacity
    predicted_labels = modelSVM.predict(val_features_std)
    accuracy_val = accuracy_score(val_labels, predicted_labels)

    # if use_proba=True, SVC can ouput calibrated probabilities with predict_proba()
    if use_proba:
        # predicted_probabilities has the shape (N, 2) = P(class=0), P(class=1)
        predicted_probabilities = modelSVM.predict_proba(val_features_std)
        prob_label1 = predicted_probabilities[:, 1] # we treat probability(label=1) as a score

        # metrics asking to be reported
        mae = np.mean(np.abs(val_labels - prob_label1))
        mse = np.mean((val_labels - prob_label1) ** 2)
        spearman_corelation, _ = spearmanr(val_labels, prob_label1)
        kendall_corelation, _ = kendalltau(val_labels, prob_label1)
    else:
        # if we don't request probabilities, we can't compute probability based metrics
        mae = mse = spearman_corelation = kendall_corelation = None

    return accuracy_train, accuracy_val, mae, mse, spearman_corelation, kendall_corelation

def main():
    # building the features for training and validation
    train_features, train_labels = feature_matrix_creator(train_dataset)
    val_features, val_labels = feature_matrix_creator(validation_dataset)

    print(train_features.shape, val_features.shape)

    # standardizing features = each feature dimension has after this the mean=0 and the std=1
    # fitting scaler on training data only and then reusing it for validation later
    scaler = StandardScaler()
    train_features_std = scaler.fit_transform(train_features)
    val_features_std = scaler.transform(val_features)

    # The GRID of hyperparameters - exploring hyperparameter space let's call it
    # C controls regularization, smaller C = more regularization, bigger C = less regularization
    # gamma controls RBF width, small gamma = smoother decision boundary, bigger gamma = more "wiggly" boundary
    lista_C = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
               0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    lista_gamma = ["scale", 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    # All the (G,gamma) pairs = basically all configurations of the hyperparameters
    lista_configurari = [(C, gamma) for C in lista_C for gamma in lista_gamma]

    rezultate = []

    def evaluate_one_configuration(C, gamma): # train and evaluate one SVM configuration
        # enable probabilities here because we want MAE/MSE/Spearman/Kendall on P(class=1)
        acc_train, acc, mae, mse, spearman_cor, kendall_cor = train_evaluate_SVM(train_features_std, train_labels,
                                                                      val_features_std, val_labels, C, gamma, use_proba=True)
        print(f"C={C}, gamma={gamma}, acc={acc:.4f}")
        return {"C": C, "gamma": gamma, "train_accuracy": acc_train, "accuracy": acc, "MAE": mae, "MSE": mse, "Spearman": spearman_cor, "Kendall": kendall_cor}

    # Parallel grid search over all configs so it goes faster on my CPU's cores - I used 16 threads here
    rezultate = Parallel(n_jobs=16, backend="loky", verbose=10)(delayed(evaluate_one_configuration)(C, gamma) for (C, gamma) in lista_configurari)

    # make the results dataframe and get it sorted by the validation accuracy
    dataframe_rezultate = pd.DataFrame(rezultate)
    dataframe_rezultate_sorted = dataframe_rezultate.sort_values(by="accuracy", ascending=False)

    # choosing the best configuration by validation accuracy
    best_row = dataframe_rezultate_sorted.iloc[0]
    best_C = best_row["C"]
    best_gamma = best_row["gamma"]

    acc_train, acc, mae, mse, spearman_cor, kendall_cor = train_evaluate_SVM(train_features_std, train_labels,
                                                                  val_features_std, val_labels, best_C, best_gamma, use_proba=True)

    best_train_accuracy = acc_train
    best_accuracy = acc
    best_mae = mae
    best_mse = mse
    best_spearman = spearman_cor
    best_kendall = kendall_cor

    # save the whole grid search in a csv that I can analize later
    csv_grid = os.path.join(HOME_FOLDER, "SVM_16_mean_pool_grid_search_results.csv")
    dataframe_rezultate_sorted.to_csv(csv_grid, index=False)

    statistics_svm = pd.DataFrame({
        "model": ["SVM_RBF_handcrafted"],
        "C": [best_C],
        "gamma": [best_gamma],
        "train_accuracy": [best_train_accuracy],
        "accuracy": [best_accuracy],
        "MAE": [best_mae],
        "MSE": [best_mse],
        "Spearman": [best_spearman],
        "Kendall": [best_kendall],
    })

    csv_best = os.path.join(HOME_FOLDER, "SVM_16_mean_pool_results_validation.csv")
    statistics_svm.to_csv(csv_best, index=False)
    print(f"\nAll Metrici salvate - Grid search - {csv_grid}")
    print(f"\nBest Metrici salvate - {csv_best}")

if __name__ == "__main__":
    main()