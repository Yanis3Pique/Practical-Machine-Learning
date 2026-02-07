# 🧠 Machine Learning Projects Portfolio

This repository contains two independent machine learning projects, both focused on **classical methods, handcrafted representations, and rigorous experimental protocols**.

- **Project 1** explores *pairwise image distribution classification* using supervised learning.
- **Project 2** explores *unsupervised visual domain separation* on real-world images.

Each project is self-contained and addresses a different learning paradigm.

---

# 📌 Project 1 - Pairwise Image Distribution Classification

This project implements a **classical machine learning approach** for a Kaggle-style problem:

> Given two noisy images, decide whether they come from the **same underlying distribution**.

The solution deliberately avoids deep learning and pretrained models, focusing instead on **explicit feature engineering** and **well-controlled classical classifiers**.

---

## 🧠 Problem Overview

- **Task:** Binary classification on *pairs of images*
- **Input:** Two grayscale images (256×256) per sample
- **Output:**
  - `1` → same distribution  
  - `0` → different distributions
- **Constraints:**
  - No pretrained models
  - No external datasets
  - No data augmentation

The main difficulty is that the images are **highly noisy**, visually similar, and lack strong global structure. Discriminative information is mostly **local and statistical**, rather than semantic.

---

## 🔍 Approach

- Explicit **handcrafted feature extraction**
- Feature aggregation at pair level (distance / difference statistics)
- Classical supervised classifiers:
  - **K-Nearest Neighbors (KNN)**
  - **Support Vector Machines (SVM)**
- Careful scaling and normalization:
  - StandardScaler
  - RobustScaler
- Systematic hyperparameter search on validation data

The emphasis is on **interpretability**, **controlled experimentation**, and understanding model behavior under noise.

---

<div align="center">✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦ ✦</div>

---


# 🐎🦓 Project 2 - Unsupervised Horse vs Zebra Separation

This project explores **unsupervised learning** for separating horses and zebras in images using **handcrafted visual features** and **clustering algorithms**, without using labels during training.

It investigates how much semantic structure can be recovered **purely from visual statistics**, without supervision.

---

## 🧠 Problem Overview

- **Dataset:** Horse2Zebra (CycleGAN)
- **Task:** Unsupervised separation of horse vs zebra images
- **Core idea:**  
  Extract visual features → cluster images → evaluate clusters *afterwards* using ground-truth labels

Guiding question:

> *Can meaningful semantic separation emerge purely from visual structure?*

---

## 🧠 Feature Representations

Two classical computer vision descriptors are evaluated.

### Histogram of Oriented Gradients (HOG)
- Captures **edges and global shape**
- Particularly effective for zebra stripe patterns

Pipeline:
- Center crop (70%)
- Grayscale conversion
- HOG extraction
- Feature standardization
- PCA dimensionality reduction (10 components)

Script: `HOG.py`

---

### Local Binary Patterns (LBP)
- Captures **local texture statistics**
- More sensitive to background noise

Pipeline:
- Center crop (70%)
- Grayscale conversion
- Uniform LBP (P = 8, R = 1)
- Normalized 10-bin histogram

Script: `LBP.py`

---

## 🔍 Unsupervised Learning Methods

### Fuzzy C-Means (FCM)
- Soft clustering with membership scores
- Hyperparameters:
  - Fuzziness parameter `m ∈ [1.05, 2.5]`
  - Distance metrics: Euclidean, Manhattan, Cosine
- **Model selection:**  
  Performed only on the training set using the **Fuzzy Partition Coefficient (FPC)**

Script: `FCM.py`

---

### Agglomerative Hierarchical Clustering (AHC)
- Bottom-up hierarchical clustering
- Hyperparameters:
  - Linkage: Ward, Average, Single, Complete
  - Distance: Euclidean or Cosine
- **Model selection:**  
  Performed only on the training set using the **silhouette score**

Since AHC has no native prediction step, test samples are assigned using the **nearest centroid** rule.

Script: `AHC.py`

---

## 📊 Baselines

### Random Baseline
- Uniform random predictions
- Averaged over **1000 trials**
- Establishes chance-level performance (~50%)

Script: `Random.py`

---

### Supervised KNN (Upper Bound)
- Uses the same HOG / LBP features
- Labels are used **only for comparison**
- Quantifies the gap between unsupervised and supervised learning

Script: `KNN.py`

---

## 🧪 Experimental Protocol

- Strict train / test separation
- Hyperparameter tuning performed **only on training data**
- Test set used **once**, for final evaluation
- Identical pipeline across all experiments

Evaluation metrics:
- Accuracy
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

---

## 📈 Results (High-Level)

- HOG features consistently outperform LBP
- Best unsupervised test accuracy: **~66.5%**
- Random baseline: **~50%**
- Supervised KNN upper bound: **~83%**
- High internal clustering scores do **not** guarantee semantic separation
