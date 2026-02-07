import os
import random
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

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
                         l1_distance, l2_distance, cosine_similarity], dtype=np.float32) # 9 scallar stats
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

def test_feature_matrix_creator(df): # same as the feature_matrix_creator, but without labels since it's used for test data
    feature_vectors = []

    for i, row in df.iterrows():
        id1 = row["id_noise_1"]
        id2 = row["id_noise_2"]

        features_vector = extract_image_pair_features(id1, id2)
        feature_vectors.append(features_vector)

    matrix_of_features = np.vstack(feature_vectors).astype(np.float32)
    return matrix_of_features

def main():
    # building the features for training and validation
    train_features, train_labels = feature_matrix_creator(train_dataset)
    val_features, val_labels = feature_matrix_creator(validation_dataset)

    # merging train + validation for more data, since we're directly predicting on the test set afterwards(more data)
    train_features = np.concatenate([train_features, val_features], axis=0)
    train_labels = np.concatenate([train_labels, val_labels], axis=0)

    print(train_features.shape, val_features.shape)

    # standardizing features = each feature dimension has after this the mean=0 and the std=1
    # fitting scaler on training data only and then reusing it for the test later
    scaler = StandardScaler()
    train_features_std = scaler.fit_transform(train_features)

    # SVM with RBF kernel
    # C controls regularization, C=0.25 -> pretty small so more regularization
    # gamma controls RBF width, gamma=0.0001 -> pretty small so smoother decision boundary
    modelSVM = SVC(kernel="rbf", C=0.25, gamma=0.0001, random_state=seed)
    modelSVM.fit(train_features_std, train_labels) # train the SVM on the standardized features

    # building the test features and applying the same scaler as from above
    test_features = test_feature_matrix_creator(test_dataset) # (4104 rows x 521 features)
    test_features_std = scaler.transform(test_features)

    # predicting the labels for the test image pairs
    test_predictions = modelSVM.predict(test_features_std)

    id_pairs = [] # here it's being saved the id_pair format required by the Kaggle competition
    for i, row in test_dataset.iterrows():
        id1 = row["id_noise_1"]
        id2 = row["id_noise_2"]
        id_pairs.append(f"({id1},{id2})")

    # saving the submission csv locally
    submission_df = pd.DataFrame({"id_pair": id_pairs, "label": test_predictions.astype(int)})
    submission_path = os.path.join(HOME_FOLDER, "sample_SVM_mean_pool_14_12_25.csv")
    submission_df.to_csv(submission_path, index=False)

if __name__ == "__main__":
    main()