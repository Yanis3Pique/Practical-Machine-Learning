import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern

random.seed(42)
np.random.seed(42)

folder_curent = os.path.dirname(os.path.abspath(__file__))
folder_dataset = os.path.join(folder_curent, "Horse2zebra")
metadatacsv = os.path.join(folder_dataset, "metadata.csv")
metadata = pd.read_csv(metadatacsv)
print("Metadata shape:", metadata.shape)
domain_labeling = {"A (Horse)": 0, "B (Zebra)": 1} # like the domain is in the csv file, but transformed to A = Horse -> 0, B = Zebra -> 1
metadata["label"] = metadata["domain"].map(domain_labeling)
absolute_paths = [] # absolute paths for images based on image_path column in the csv
for i, row in metadata.iterrows():
    full_path = os.path.join(folder_dataset, row["image_path"])
    absolute_paths.append(full_path)
metadata["absolute_path"] = absolute_paths


def count_classes(labels): # just for printing in the console so it's a bit clear what we work with
    nr_horses, nr_zebras = 0, 0
    for label in labels:
        if label == 0: nr_horses += 1
        else: nr_zebras += 1
    return {"horses": nr_horses, "zebras": nr_zebras}


train_df = metadata[metadata["split"] == "train"].reset_index(drop=True) # split images into train and test
test_df = metadata[metadata["split"] == "test"].reset_index(drop=True)
train_paths = train_df["absolute_path"].tolist() # train paths and labels
train_labels = train_df["label"].astype(int).tolist()
test_paths = test_df["absolute_path"].tolist() # test paths and labels
test_labels = test_df["label"].astype(int).tolist()
print("Train images:", len(train_paths))
print("Test images :", len(test_paths))
print("Train distribution:", count_classes(train_labels))
print("Test distribution:", count_classes(test_labels))


def loading_image(path): # images are already 256x256 in this dataset
    imagine = Image.open(path).convert("RGB")
    nparray = np.asarray(imagine).astype(np.float32) / 255.0 # normalizing in [0, 1]
    return nparray # (height=256, width=256, 3)


def center_crop(rgb, crop_ratio=0.70): # cropping the center because the animal is usually more centered than the background
    h, w, _ = rgb.shape
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    y = (h - crop_h) // 2
    x = (w - crop_w) // 2
    return rgb[y:y + crop_h, x:x + crop_w, :] # from 256x256 we get 179x179


def lbp_features(rgb_crop): # local binary patterns -> texture descriptor
    # manual RGB to greyscale
    grey = (0.299 * rgb_crop[:, :, 0] + 0.587 * rgb_crop[:, :, 1] + 0.114 * rgb_crop[:, :, 2]).astype(np.float32)
    P = 8 # neighbors nr
    R = 1 # radius
    method = "uniform" # uniform LBP -> smaller, more stable histogram
    lbp = local_binary_pattern(grey, P=P, R=R, method=method)

    nr_bins = P + 2 # nr bins for uniform LBP = P + 2 (8 edge like + 2 flat)

    histograma, _ = np.histogram(
        lbp.flatten(),
        bins=nr_bins,
        range=(0, nr_bins), # (0, 10) in our case -> 0 for flat(just 0s), 1-8 for edge/line patterns, 9 for non-uniform(noisy) patterns
        density=False
    )
    histograma = histograma.astype(np.float32)
    histograma = histograma / (histograma.sum() + 1e-12) # normalize so histogram sums to 1

    return histograma # shape (P + 2,)


def extract_features_for_one_image(path, crop_ratio=0.70):
    imagine_rgb = loading_image(path) # (256, 256, 3)
    imagine_rgb_crop = center_crop(imagine_rgb, crop_ratio=crop_ratio)
    features_lbp = lbp_features(imagine_rgb_crop)
    return features_lbp


def build_feature_matrix(image_paths, crop_ratio=0.70):
    lista_features = []
    for path in image_paths:
        features_lbp = extract_features_for_one_image(path, crop_ratio=crop_ratio)
        lista_features.append(features_lbp)
    return np.vstack(lista_features).astype(np.float32)


def main():
    crop_ratio = 0.70 # keep the center 70% so we kinda reduce the background noise

    print()
    print("Extract handcrafted features (LBP)")
    train_features = build_feature_matrix(train_paths, crop_ratio=crop_ratio)
    test_features  = build_feature_matrix(test_paths, crop_ratio=crop_ratio)

    print("train_features:", train_features.shape)
    print("test_features:", test_features.shape)

    out_dir = os.path.join(folder_curent, "features_lbp")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "train_lbp_features.npy"), train_features)
    np.save(os.path.join(out_dir, "test_lbp_features.npy"), test_features)
    np.save(os.path.join(out_dir, "train_labels.npy"), np.array(train_labels, dtype=np.int64))
    np.save(os.path.join(out_dir, "test_labels.npy"), np.array(test_labels, dtype=np.int64))

    print()
    print("Saved in:", out_dir)
    print("train_features:", train_features.shape)
    print("test_features:", test_features.shape)


if __name__ == "__main__":
    main()