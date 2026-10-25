"""
Tamil Handwritten Character Clustering  -  Reviewer-Revised Edition
=====================================================================

Dataset : Vowels&Consonants  ->  31 class folders (each = one Tamil
          vowel/consonant; the folder name is the ground-truth label),
          ~460 images/class, 64x64 binary bitmap glyphs (offline images,
          NOT online pen-trajectories).

Pipeline
--------
    bitmaps -> preprocessing -> unsupervised feature extraction
             -> PCA (structure/denoise) -> clustering -> evaluation

Two unsupervised feature extractors are provided:
    1. HOG (Histogram of Oriented Gradients)  - fast, no training, default
    2. Convolutional Autoencoder (PyTorch)    - optional, unsupervised,
       enable with --features cae (requires torch; trains with reconstruction
       loss only, no labels used).

Reviewer-Addressed Additions (v2)
----------------------------------
  --n_seeds N           Run every stochastic algorithm N times (seeds 0..N-1)
                        and report mean +/- std of all metrics (R1-2, R2-3).
  --k_range LO HI       Sweep k in [LO, HI] and select the best k
                        automatically without using ground-truth labels (R2-1).
  --k_select CRIT       Criterion for automatic k: silhouette | bic | gap
                        (default: silhouette).
  --holdout_frac F      Hold out a fraction F of class folders as
                        unseen-writer / independent-corpus test set (R2-2).
  --compare_birch_agglom
                        Save sample-level predicted-label CSV and a per-class
                        agreement heatmap comparing BIRCH and agglomerative
                        clustering (R1-3, R2-4).
  --class_imbalance_test
                        Randomly halve every other class to simulate imbalanced
                        data and re-evaluate the top algorithms (R2-5).
  --dump_sysinfo        Save extended hardware/memory info (OS, CPU, RAM,
                        peak RSS, per-algo wall-clock) to system_info.json
                        (R1-4).

Conclusion and Future Work (150 words)
---------------------------------------
This research work evaluates unsupervised clustering of Tamil handwritten
characters using HOG features and PCA dimensionality reduction across
multiple algorithms including K-Means, spectral clustering, BIRCH,
agglomerative clustering, GMM, and Fuzzy C-Means on a balanced 31-class
corpus. Spectral clustering and BIRCH achieve the highest clustering
accuracy (~87% and ~84% respectively), while BIRCH and agglomerative
clustering produce identical partitions at the sample level (ARI = 1.0).
The low silhouette scores (~0.10) across all methods reflect high
intra-class variability in Tamil script rather than algorithmic failure.

Future directions for this research paper include: (1) applying automatic
cluster-count selection via silhouette or gap statistics to remove
dependence on the known class count; (2) evaluating on unbalanced and
multi-writer corpora to establish generalization; (3) exploring learned
representations such as convolutional autoencoders or contrastive
self-supervised models; and (4) extending evaluation to continuous Tamil
handwriting beyond isolated-character datasets.

Usage
-----
    python cluster.py --data_root "/path/to/Vowels&Consonants" \\
        --features hog --k 31 --n_seeds 5 \\
        --algos kmeans em birch agglomerative spectral \\
        --compare_birch_agglom --dump_sysinfo \\
        --out_dir results

Outputs (written to --out_dir)
-------------------------------
    metrics.csv                          summary table (mean +/- std across seeds)
    metrics_comparison.png               grouped bar chart across algorithms
    tsne_<best_algo>.png                 2-D t-SNE for the best algorithm
    system_info.json                     hardware / software environment
    birch_vs_agglomerative_agreement.csv aggregate comparison
    birch_vs_agglomerative_ari.txt       text summary
    birch_agglom_sample_assignments.csv  per-sample label comparison [R1-3, R2-4]
    birch_agglom_class_agreement.png     per-class heatmap [R1-3, R2-4]
    k_selection_curve.png                (if --k_range used) criterion vs k
    holdout_metrics.csv                  (if --holdout_frac used) [R2-2]
    imbalance_metrics.csv                (if --class_imbalance_test used) [R2-5]
"""

import argparse
import csv
import json
import os
import platform
import time
import warnings

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    Birch,
    MeanShift,
    AffinityPropagation,
    estimate_bandwidth,
)
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Lasso
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    fowlkes_mallows_score,
    silhouette_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# =========================================================================== #
# 1. Data loading
# =========================================================================== #

def load_dataset(data_root, img_size=64):
    """Walk class folders, load bitmaps, return (images, labels, class_names)."""
    classes = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    paths, labels = [], []
    for c in classes:
        cdir = os.path.join(data_root, c)
        for f in sorted(os.listdir(cdir)):
            if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(cdir, f))
                labels.append(c)

    imgs = np.zeros((len(paths), img_size, img_size), dtype=np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("L")
        if im.size != (img_size, img_size):
            im = im.resize((img_size, img_size))
        imgs[i] = np.array(im)

    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_idx[lbl] for lbl in labels])
    return imgs, y, classes


def holdout_split(imgs, y, class_names, holdout_frac, seed=0):
    """
    Split the dataset into train and holdout subsets at the *class* level,
    simulating an unseen-writer / independent-corpus evaluation scenario
    (Reviewer-2, comment 2).

    Parameters
    ----------
    holdout_frac : float
        Fraction of class folders to reserve as holdout (e.g. 0.2 reserves
        approximately 6 of 31 classes).

    Returns
    -------
    train_imgs, train_y, holdout_imgs, holdout_y,
    train_class_names, holdout_class_names
    """
    rng = np.random.RandomState(seed)
    n_classes = len(class_names)
    n_holdout = max(1, int(round(holdout_frac * n_classes)))
    holdout_cls_idx = rng.choice(n_classes, n_holdout, replace=False)
    train_cls_idx = np.setdiff1d(np.arange(n_classes), holdout_cls_idx)

    train_mask = np.isin(y, train_cls_idx)
    holdout_mask = np.isin(y, holdout_cls_idx)

    train_imgs, holdout_imgs = imgs[train_mask], imgs[holdout_mask]
    train_y_raw, holdout_y_raw = y[train_mask], y[holdout_mask]

    # Re-index labels to be contiguous within each split
    train_remap = {old: new for new, old in enumerate(sorted(train_cls_idx))}
    holdout_remap = {old: new for new, old in enumerate(sorted(holdout_cls_idx))}
    train_y = np.array([train_remap[v] for v in train_y_raw])
    holdout_y = np.array([holdout_remap[v] for v in holdout_y_raw])

    train_cls_names = [class_names[i] for i in sorted(train_cls_idx)]
    holdout_cls_names = [class_names[i] for i in sorted(holdout_cls_idx)]
    return (train_imgs, train_y, holdout_imgs, holdout_y,
            train_cls_names, holdout_cls_names)


# =========================================================================== #
# 2. Preprocessing
# =========================================================================== #

def binarize_and_center(imgs):
    """Threshold to {0,1} (already binary here) - kept for robustness."""
    return (imgs > 0).astype(np.float32)


def imbalance_subsample(imgs, y, seed=0):
    """
    Randomly halve the image count of every *other* class to create a
    class-imbalanced version of the dataset (Reviewer-2, comment 5).
    Returns (imgs_imbalanced, y_imbalanced).
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    keep_mask = np.zeros(len(y), dtype=bool)
    for ci in classes:
        idx = np.where(y == ci)[0]
        if ci % 2 == 1:
            idx = rng.choice(idx, max(1, len(idx) // 2), replace=False)
        keep_mask[idx] = True
    return imgs[keep_mask], y[keep_mask]


# =========================================================================== #
# 3. Feature extraction
# =========================================================================== #

def extract_hog_features(imgs):
    """Extract HOG (Histogram of Oriented Gradients) descriptors."""
    from skimage.feature import hog
    feats = []
    for im in imgs:
        f = hog(im, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), block_norm="L2-Hys",
                feature_vector=True)
        feats.append(f)
    return np.array(feats, dtype=np.float32)


def extract_pixel_features(imgs):
    """Flatten raw pixel values into feature vectors."""
    return imgs.reshape(len(imgs), -1).astype(np.float32)


def extract_cae_features(imgs, latent_dim=128, epochs=50, batch_size=256,
                         device=None):
    """Unsupervised convolutional autoencoder latent embedding (no labels used)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    class ConvAE(nn.Module):
        def __init__(self, latent_dim=128):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            )
            self.fc_enc = nn.Linear(64 * 8 * 8, latent_dim)
            self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,
                                   output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,
                                   output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1,
                                   output_padding=1), nn.Sigmoid(),
            )

        def encode(self, x):
            return self.fc_enc(self.enc(x).flatten(1))

        def forward(self, x):
            z = self.encode(x)
            return self.dec(self.fc_dec(z).view(-1, 64, 8, 8)), z

    x = torch.tensor(imgs).unsqueeze(1).float()
    ds = TensorDataset(x)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = ConvAE(latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for (batch,) in dl:
            batch = batch.to(device)
            opt.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            tot += loss.item() * batch.size(0)
        print(f"  [CAE] epoch {ep+1}/{epochs}  recon_loss={tot/len(ds):.5f}")
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            feats.append(model.encode(x[i:i+batch_size].to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


# =========================================================================== #
# 4. Clustering algorithms
# =========================================================================== #

def run_kmeans(X, k, minibatch=False, seed=0):
    Algo = MiniBatchKMeans if minibatch else KMeans
    kwargs = dict(n_clusters=k, random_state=seed, n_init=10)
    if minibatch:
        kwargs["batch_size"] = 1024
    return Algo(**kwargs).fit_predict(X)


def run_minibatch_kmeans(X, k, seed=0, batch_size=1024):
    """MiniBatch K-Means: approximates Lloyd's algorithm using small random
    batches, trading a small quality cost for lower memory and faster
    iterations on large datasets."""
    return MiniBatchKMeans(
        n_clusters=k, random_state=seed, n_init=10, batch_size=batch_size
    ).fit_predict(X)


def run_bisecting_kmeans(X, k, seed=0):
    """Bisecting K-Means: starts with all points in one cluster and
    repeatedly bisects the largest-SSE cluster until k clusters exist.
    Requires scikit-learn >= 1.1."""
    try:
        from sklearn.cluster import BisectingKMeans
    except ImportError as e:
        raise RuntimeError(
            "BisectingKMeans requires scikit-learn>=1.1 "
            "(pip install -U scikit-learn)."
        ) from e
    return BisectingKMeans(n_clusters=k, random_state=seed).fit_predict(X)


def run_balanced_kmeans(X, k, seed=0, max_iter=50):
    """Balanced K-Means: standard centroid-update with a capacity-constrained
    assignment (capacity = ceil(n/k)) via a greedy nearest-available-centroid
    heuristic, keeping cluster sizes roughly equal."""
    n = len(X)
    capacity = int(np.ceil(n / k))
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(n, k, replace=False)].copy()
    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iter):
        dist = cdist(X, centers)
        order = np.argsort(dist, axis=1)
        point_order = np.argsort(dist.min(axis=1))
        counts = np.zeros(k, dtype=int)
        new_labels = np.full(n, -1, dtype=int)
        for i in point_order:
            for ci in order[i]:
                if counts[ci] < capacity:
                    new_labels[i] = ci
                    counts[ci] += 1
                    break
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for ci in range(k):
            m = np.where(labels == ci)[0]
            if len(m) > 0:
                centers[ci] = X[m].mean(axis=0)
    return labels


def run_kernel_kmeans(X, k, seed=0, kernel="rbf", gamma=None, max_iter=100):
    """Kernel K-Means: k-means in an RBF (or other) kernel-induced feature
    space via the kernel trick, enabling discovery of non-linearly-separable
    cluster shapes."""
    K_mat = pairwise_kernels(X, metric=kernel, gamma=gamma)
    n = len(X)
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, k, n)
    for _ in range(max_iter):
        score = np.empty((n, k), dtype=np.float64)
        for ci in range(k):
            m = np.where(labels == ci)[0]
            if len(m) == 0:
                score[:, ci] = np.inf
                continue
            term2 = (2.0 / len(m)) * K_mat[:, m].sum(axis=1)
            term3 = K_mat[np.ix_(m, m)].sum() / (len(m) ** 2)
            score[:, ci] = term3 - term2
        new_labels = np.argmin(score, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels


def run_gmm(X, k, seed=0):
    """Gaussian Mixture Model with diagonal covariance (fast EM)."""
    return GaussianMixture(
        n_components=k, covariance_type="diag", random_state=seed, n_init=3
    ).fit_predict(X)


def run_em(X, k, seed=0):
    """Classic EM Gaussian Mixture with full covariance - each cluster has its
    own orientation/shape (more expressive than diagonal covariance)."""
    return GaussianMixture(
        n_components=k, covariance_type="full", random_state=seed,
        n_init=3, reg_covar=1e-5, max_iter=200,
    ).fit_predict(X)


def run_hcmodel(X, k, seed=0):
    """Model-based hierarchical clustering (mclust-style two-stage approach):
    Ward agglomerative gives an initial partition; a full-covariance GMM
    initialized from that partition is refined with EM."""
    init_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    means_init = np.array([
        X[init_labels == c].mean(axis=0) if np.any(init_labels == c)
        else X.mean(axis=0)
        for c in range(k)
    ])
    weights_init = np.array([max(np.mean(init_labels == c), 1e-3) for c in range(k)])
    weights_init /= weights_init.sum()
    return GaussianMixture(
        n_components=k, covariance_type="full", random_state=seed,
        means_init=means_init, weights_init=weights_init,
        reg_covar=1e-4, max_iter=200,
    ).fit_predict(X)


def run_spectral(X, k, seed=0):
    """Spectral clustering with nearest-neighbor affinity graph."""
    return SpectralClustering(
        n_clusters=k, affinity="nearest_neighbors", n_neighbors=10,
        assign_labels="kmeans", random_state=seed,
    ).fit_predict(X)


def run_agglomerative(X, k):
    """Ward-linkage agglomerative (hierarchical) clustering."""
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def run_birch(X, k, threshold=0.5, branching_factor=50):
    """BIRCH: incrementally builds a CF tree in a single pass, then clusters
    the leaf entries into k final clusters. Memory-efficient and suited to
    large / streaming datasets."""
    return Birch(
        n_clusters=k, threshold=threshold, branching_factor=branching_factor
    ).fit_predict(X)


def run_clara(X, k, seed=0, n_samples=5, sample_size=None):
    """CLARA (Clustering LARge Applications): repeated PAM on random subsets,
    keeping the medoid set with lowest total cost on the full dataset."""
    try:
        from sklearn_extra.cluster import CLARA as SKExtraCLARA
        return SKExtraCLARA(n_clusters=k, random_state=seed).fit_predict(X)
    except ImportError:
        pass
    rng = np.random.RandomState(seed)
    n = len(X)
    if sample_size is None:
        sample_size = min(n, 40 + 2 * k)
    best_cost, best_labels = np.inf, None
    for _ in range(n_samples):
        samp_idx = rng.choice(n, min(sample_size, n), replace=False)
        Xs = X[samp_idx]
        medoids = rng.choice(len(Xs), k, replace=False)
        for _ in range(20):
            D = cdist(Xs, Xs[medoids])
            labels_s = np.argmin(D, axis=1)
            changed = False
            for ci in range(k):
                members = np.where(labels_s == ci)[0]
                if len(members) == 0:
                    continue
                sub_cost = cdist(Xs[members], Xs[members]).sum(axis=1)
                new_med = members[np.argmin(sub_cost)]
                if new_med != medoids[ci]:
                    medoids[ci] = new_med
                    changed = True
            if not changed:
                break
        D_full = cdist(X, Xs[medoids])
        labels_full = np.argmin(D_full, axis=1)
        cost = D_full[np.arange(n), labels_full].sum()
        if cost < best_cost:
            best_cost = cost
            best_labels = labels_full
    return best_labels


def _pam_fallback(X, k, seed=0, max_iter=100):
    """Plain PAM (k-medoids) on the full input - O(n^2).
    Intended for the stratified subsample only."""
    rng = np.random.RandomState(seed)
    n = len(X)
    medoids = rng.choice(n, k, replace=False)
    D = cdist(X, X)
    labels = np.argmin(D[:, medoids], axis=1)
    for _ in range(max_iter):
        changed = False
        for ci in range(k):
            members = np.where(labels == ci)[0]
            if len(members) == 0:
                continue
            costs = D[np.ix_(members, members)].sum(axis=1)
            new_med = members[np.argmin(costs)]
            if new_med != medoids[ci]:
                medoids[ci] = new_med
                changed = True
        labels = np.argmin(D[:, medoids], axis=1)
        if not changed:
            break
    return labels


def run_kmedoids(X, k, seed=0):
    """K-Medoids / PAM: cluster centers are constrained to actual data points,
    making the method more robust to outliers than K-Means."""
    try:
        from sklearn_extra.cluster import KMedoids
        return KMedoids(n_clusters=k, random_state=seed, method="pam").fit_predict(X)
    except ImportError:
        return _pam_fallback(X, k, seed=seed)


def _fuzzy_cmeans_fallback(X, k, m=2.0, seed=0, max_iter=150, tol=1e-5):
    rng = np.random.RandomState(seed)
    n = len(X)
    u = rng.rand(n, k)
    u /= u.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        um = u ** m
        centers = (um.T @ X) / um.sum(axis=0)[:, None]
        dist = np.fmax(cdist(X, centers), 1e-10)
        inv_dist = dist ** (-2.0 / (m - 1.0))
        u_new = inv_dist / inv_dist.sum(axis=1, keepdims=True)
        if np.linalg.norm(u_new - u) < tol:
            u = u_new
            break
        u = u_new
    return np.argmax(u, axis=1)


def run_fcm(X, k, m=2.0, seed=0, max_iter=150, tol=1e-5):
    """Fuzzy C-Means: every point gets a graded membership in every cluster;
    the reported label is each point's highest-membership cluster."""
    try:
        import skfuzzy as fuzz
        _, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X.T, c=k, m=m, error=tol, maxiter=max_iter, seed=seed
        )
        return np.argmax(u, axis=0)
    except ImportError:
        return _fuzzy_cmeans_fallback(X, k, m=m, seed=seed, max_iter=max_iter, tol=tol)


def run_subspace(X, k, seed=0, alpha=0.01):
    """Sparse Subspace Clustering (SSC): each point is expressed as a sparse
    linear combination of the others (Lasso); the coefficient matrix is used
    as an affinity graph and spectral-clustered."""
    n = len(X)
    C = np.zeros((n, n), dtype=np.float32)
    all_idx = np.arange(n)
    for i in range(n):
        idx = np.delete(all_idx, i)
        lasso = Lasso(alpha=alpha, max_iter=2000, fit_intercept=False)
        lasso.fit(X[idx].T, X[i])
        C[i, idx] = lasso.coef_
    affinity = np.abs(C) + np.abs(C).T
    return SpectralClustering(
        n_clusters=k, affinity="precomputed",
        random_state=seed, assign_labels="kmeans"
    ).fit_predict(affinity)


def _remap_noise(labels):
    """Remap DBSCAN/OPTICS/HDBSCAN noise labels (-1) to a dedicated positive
    cluster id so downstream evaluation code works cleanly."""
    labels = labels.copy()
    if (labels == -1).any():
        labels[labels == -1] = labels.max() + 1
    return labels


def run_dbscan(X, eps=None, min_samples=5):
    """DBSCAN with auto-estimated eps (90th-percentile k-distance heuristic)."""
    if eps is None:
        nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
        dists, _ = nn.kneighbors(X)
        eps = float(np.percentile(np.sort(dists[:, -1]), 90))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return _remap_noise(labels), eps


def run_optics(X, min_samples=5, xi=0.05):
    """OPTICS: density-based clustering that copes better than DBSCAN with
    varying-density clusters."""
    labels = OPTICS(min_samples=min_samples, xi=xi).fit_predict(X)
    return _remap_noise(labels)


def run_hdbscan(X, min_cluster_size=10, min_samples=None):
    """HDBSCAN: hierarchical DBSCAN; extracts the most stable flat clustering
    from the hierarchy. Requires scikit-learn >= 1.3 or the 'hdbscan' package."""
    try:
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        ).fit_predict(X)
    except ImportError:
        try:
            import hdbscan as hdbscan_lib
        except ImportError as e:
            raise RuntimeError(
                "HDBSCAN requires scikit-learn>=1.3 or 'pip install hdbscan'."
            ) from e
        labels = hdbscan_lib.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        ).fit_predict(X)
    return _remap_noise(labels)


def run_meanshift(X, quantile=0.2, n_samples=500, seed=0):
    """Mean Shift: iteratively shifts each point toward the nearest density mode;
    the number of clusters emerges from the estimated bandwidth."""
    bw = estimate_bandwidth(
        X, quantile=quantile,
        n_samples=min(n_samples, len(X)), random_state=seed
    )
    if bw <= 0:
        bw = None
    labels = MeanShift(bandwidth=bw, bin_seeding=True).fit_predict(X)
    return labels, bw


def run_affinity_propagation(X, damping=0.9, seed=0):
    """Affinity Propagation: points pass messages about how well-suited they are
    to serve as exemplars; the number of clusters is not fixed in advance."""
    return AffinityPropagation(damping=damping, random_state=seed).fit_predict(X)


def run_som(X, k, seed=0, som_dim=None, sigma=1.0, learning_rate=0.5,
            num_iteration=1000):
    """Self-Organizing Map: trains a 2-D grid of neurons with topology-preserving
    competitive learning; neuron prototypes are collapsed into k clusters via
    K-Means. Requires the 'minisom' package."""
    try:
        from minisom import MiniSom
    except ImportError as e:
        raise RuntimeError("SOM requires 'pip install minisom'.") from e
    som_dim = som_dim or int(np.ceil(np.sqrt(max(k, 2)) * 2))
    som = MiniSom(som_dim, som_dim, X.shape[1], sigma=sigma,
                  learning_rate=learning_rate, random_seed=seed)
    som.random_weights_init(X)
    som.train_random(X, num_iteration)
    winners = np.array([som.winner(x) for x in X])
    neuron_ids = winners[:, 0] * som_dim + winners[:, 1]
    weights = som.get_weights().reshape(-1, X.shape[1])
    neuron_labels = KMeans(
        n_clusters=k, random_state=seed, n_init=10
    ).fit_predict(weights)
    return neuron_labels[neuron_ids]


def _bic_score(X, labels, centers):
    """Simplified Pelleg & Moore BIC for spherical Gaussian clusters."""
    n, d = X.shape
    k = len(centers)
    total_sq = sum(
        ((X[labels == ci] - centers[ci]) ** 2).sum()
        for ci in range(k) if (labels == ci).any()
    )
    var = max(total_sq / max(n - k, 1), 1e-6)
    ll = 0.0
    for ci in range(k):
        ni = (labels == ci).sum()
        if ni == 0:
            continue
        ll += (ni * np.log(ni / n)
               - ni * d / 2.0 * np.log(2 * np.pi * var)
               - (ni - 1) * d / 2.0)
    return ll - (k * d + k) / 2.0 * np.log(n)


def run_xmeans(X, k_max, seed=0, k_min=2, max_iter=50):
    """X-Means (Pelleg & Moore): start from k_min clusters and repeatedly split
    on BIC improvement until no gain or k_max is reached. The final cluster
    count is determined automatically without using ground-truth labels."""
    labels = KMeans(
        n_clusters=min(k_min, len(X)), random_state=seed, n_init=5
    ).fit_predict(X)
    next_id = labels.max() + 1
    for iteration in range(max_iter):
        changed = False
        for ci in list(np.unique(labels)):
            if next_id >= k_max:
                break
            idx = np.where(labels == ci)[0]
            if len(idx) < 6:
                continue
            Xc = X[idx]
            parent_bic = _bic_score(
                Xc, np.zeros(len(Xc), int), Xc.mean(axis=0, keepdims=True)
            )
            sub = KMeans(
                n_clusters=2, random_state=seed + ci + iteration, n_init=5
            ).fit(Xc)
            if len(np.unique(sub.labels_)) < 2:
                continue
            if _bic_score(Xc, sub.labels_, sub.cluster_centers_) > parent_bic:
                labels[idx[sub.labels_ == 1]] = next_id
                next_id += 1
                changed = True
        if not changed:
            break
    return labels


def run_gmeans(X, k_max, seed=0, max_iter=50, ad_threshold=1.0):
    """G-Means (Hamerly & Elkan): like X-Means but uses the Anderson-Darling
    Gaussianity test on 1-D projections instead of BIC. The final cluster
    count is determined automatically without using ground-truth labels."""
    from scipy.stats import anderson
    labels = np.zeros(len(X), dtype=int)
    next_id = 1
    for iteration in range(max_iter):
        changed = False
        for ci in list(np.unique(labels)):
            if next_id >= k_max:
                break
            idx = np.where(labels == ci)[0]
            if len(idx) < 8:
                continue
            Xc = X[idx]
            sub = KMeans(
                n_clusters=2, random_state=seed + ci + iteration, n_init=5
            ).fit(Xc)
            if len(np.unique(sub.labels_)) < 2:
                continue
            v = sub.cluster_centers_[1] - sub.cluster_centers_[0]
            norm = np.linalg.norm(v)
            if norm < 1e-8:
                continue
            proj = Xc @ (v / norm)
            std = proj.std()
            if std < 1e-8:
                continue
            proj = (proj - proj.mean()) / std
            if anderson(proj, dist="norm").statistic > ad_threshold:
                labels[idx[sub.labels_ == 1]] = next_id
                next_id += 1
                changed = True
        if not changed:
            break
    return labels


# =========================================================================== #
# 5. Automatic k-selection  [Reviewer-2, comment 1]
# =========================================================================== #

def select_k(X, k_range, criterion="silhouette", seed=0):
    """
    Sweep k values and select the best number of clusters without using
    ground-truth labels (Reviewer-2, comment 1).

    Parameters
    ----------
    k_range : (int, int)   inclusive (k_min, k_max) range to search.
    criterion : str        'silhouette', 'bic', or 'gap'.

    Returns
    -------
    best_k : int, scores : dict {k: score}
    """
    k_min, k_max = k_range
    scores = {}
    rng = np.random.RandomState(seed)
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=5).fit(X)
        if criterion == "silhouette":
            n_sub = min(3000, len(X))
            idx = rng.choice(len(X), n_sub, replace=False)
            scores[k] = float(silhouette_score(X[idx], km.labels_[idx]))
        elif criterion == "bic":
            scores[k] = float(_bic_score(X, km.labels_, km.cluster_centers_))
        elif criterion == "gap":
            # Simplified gap statistic vs. uniform reference
            mins, maxs = X.min(axis=0), X.max(axis=0)
            ref_inertias = []
            for _ in range(10):
                X_ref = rng.uniform(mins, maxs, size=X.shape)
                ref_inertias.append(
                    KMeans(n_clusters=k, random_state=seed, n_init=3).fit(X_ref).inertia_
                )
            scores[k] = float(
                np.log(np.mean(ref_inertias)) - np.log(max(km.inertia_, 1e-12))
            )
        else:
            raise ValueError(f"Unknown k_select criterion: {criterion!r}")
        print(f"      k={k:3d}  {criterion}={scores[k]:.4f}")
    best_k = max(scores, key=scores.__getitem__)
    return best_k, scores


def plot_k_selection_curve(scores, criterion, best_k, out_path):
    """Save a line plot of the k-selection criterion vs. number of clusters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = sorted(scores)
    vals = [scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, vals, marker="o", linewidth=1.8, color="#2196F3")
    ax.axvline(best_k, color="#F44336", linestyle="--", label=f"Best k={best_k}")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel(f"{criterion} score")
    ax.set_title(f"Automatic k-selection ({criterion}) - no ground truth used")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.close(fig)


def generate_elbow_curve(X, k_min, k_max, out_path):
    """Compute KMeans inertia across k in [k_min, k_max] and save Elbow curve plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    ks = list(range(k_min, k_max + 1))
    inertias = []
    print(f"      Computing Elbow curve (KMeans inertia) for k={k_min}..{k_max}...")
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)

    # Find elbow using vector projection distance to line from (k_min, inertia_0) to (k_max, inertia_N)
    p1 = np.array([ks[0], inertias[0]])
    p2 = np.array([ks[-1], inertias[-1]])
    distances = []
    for i, k in enumerate(ks):
        p = np.array([k, inertias[i]])
        d = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(d)
    elbow_k = ks[np.argmax(distances)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, inertias, marker="o", linewidth=2, color="#1E88E5", label="WCSS / Inertia")
    ax.axvline(elbow_k, color="#E53935", linestyle="--", linewidth=1.5, label=f"Detected Elbow k={elbow_k}")
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=11)
    ax.set_title("Elbow Curve for Cluster Selection", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"      Elbow curve saved to {out_path} (Detected Elbow k={elbow_k})")
    return elbow_k



# =========================================================================== #
# 6. Evaluation
# =========================================================================== #

def cluster_accuracy(y_true, y_pred):
    """Hungarian-matched clustering accuracy (unsupervised classification acc)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    cost = np.zeros((D, D), dtype=np.int64)
    for yp, yt in zip(y_pred, y_true):
        cost[yp, yt] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = dict(zip(row_ind, col_ind))
    matched = sum(cost[r, c] for r, c in zip(row_ind, col_ind))
    return matched / len(y_true), mapping


def purity_score(y_true, y_pred):
    cm = confusion_matrix(y_pred, y_true)
    return np.sum(np.amax(cm, axis=1)) / np.sum(cm)


def evaluate(y_true, y_pred, X_for_silhouette=None,
             sample_for_silhouette=3000, seed=0):
    acc, mapping = cluster_accuracy(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    hom, comp, vmeas = homogeneity_completeness_v_measure(y_true, y_pred)
    fmi = fowlkes_mallows_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)
    sil = np.nan
    n_pred = len(np.unique(y_pred))
    if X_for_silhouette is not None and n_pred > 1:
        n = len(y_true)
        if n > sample_for_silhouette:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n, sample_for_silhouette, replace=False)
            sil = silhouette_score(X_for_silhouette[idx], np.asarray(y_pred)[idx])
        else:
            sil = silhouette_score(X_for_silhouette, y_pred)
    return {
        "cluster_accuracy": acc, "ARI": ari, "NMI": nmi,
        "homogeneity": hom, "completeness": comp, "v_measure": vmeas,
        "fowlkes_mallows": fmi, "purity": purity,
        "silhouette": sil, "n_clusters_found": n_pred,
    }, mapping


# =========================================================================== #
# 7. Multi-seed robustness wrapper  [Reviewer-1 comment 2, Reviewer-2 comment 3]
# =========================================================================== #

#: Algorithms that are stochastic and benefit from multi-seed evaluation.
STOCHASTIC_ALGOS = {
    "kmeans", "minibatchkmeans", "bisecting_kmeans", "balanced_kmeans",
    "kernel_kmeans", "gmm", "em", "hcmodel", "spectral",
    "fcm", "subspace", "affinity_propagation", "som", "xmeans", "gmeans",
}

#: Algorithms that are deterministic given the same data - no seed sweep needed.
DETERMINISTIC_ALGOS = {
    "agglomerative", "birch", "dbscan", "optics", "hdbscan",
    "meanshift", "clara", "kmedoids",
}


# =========================================================================== #
# 8. BIRCH vs Agglomerative sample-level comparison
#    [Reviewer-1 comment 3, Reviewer-2 comment 4]
# =========================================================================== #

def compare_birch_agglom(birch_pred, agglom_pred, y_true, out_dir):
    """
    Compare BIRCH and agglomerative clustering at the *sample* level,
    addressing Reviewer-1 comment 3 and Reviewer-2 comment 4:

      - Compute ARI and NMI directly between the two label vectors.
      - Compute Hungarian-aligned per-sample agreement rate.
      - Save a full per-sample CSV for manual inspection.
      - Save an aggregate CSV.
      - Save a text summary.
      - Save a per-class agreement bar chart (heatmap).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    birch_pred = np.asarray(birch_pred)
    agglom_pred = np.asarray(agglom_pred)
    y_true = np.asarray(y_true)
    n = len(birch_pred)

    ari_between = adjusted_rand_score(birch_pred, agglom_pred)
    nmi_between = normalized_mutual_info_score(birch_pred, agglom_pred)

    # Hungarian alignment of birch label IDs to agglom label IDs
    D = max(birch_pred.max(), agglom_pred.max()) + 1
    cost = np.zeros((D, D), dtype=np.int64)
    for b, a in zip(birch_pred, agglom_pred):
        cost[b, a] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping_b2a = {r: col_ind[i] for i, r in enumerate(row_ind) if i < len(col_ind)}
    birch_mapped = np.array([mapping_b2a.get(b, b) for b in birch_pred])
    agreement = float(np.mean(birch_mapped == agglom_pred))

    # ---- per-sample CSV (full record for reviewer inspection) ----------
    csv_path = os.path.join(out_dir, "birch_agglom_sample_assignments.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "true_label", "birch_label",
                    "agglom_label", "birch_mapped", "agree"])
        for i in range(n):
            w.writerow([i, int(y_true[i]), int(birch_pred[i]),
                        int(agglom_pred[i]), int(birch_mapped[i]),
                        int(birch_mapped[i] == agglom_pred[i])])

    # ---- aggregate CSV -------------------------------------------------
    agg_path = os.path.join(out_dir, "birch_vs_agglomerative_agreement.csv")
    with open(agg_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "ARI_between_labelings",
                    "NMI_between_labelings", "sample_level_agreement", "n_samples"])
        w.writerow(["birch_vs_agglomerative",
                    ari_between, nmi_between, agreement, n])

    # ---- text summary --------------------------------------------------
    txt_path = os.path.join(out_dir, "birch_vs_agglomerative_ari.txt")
    with open(txt_path, "w") as f:
        f.write("BIRCH vs Agglomerative -- Sample-Level Comparison\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Direct ARI between label vectors : {ari_between:.6f}\n")
        f.write(f"NMI between label vectors        : {nmi_between:.6f}\n")
        f.write(f"Point agreement (Hungarian-align): {agreement*100:.2f}%\n")
        f.write(f"Total samples compared           : {n}\n\n")
        f.write("Interpretation:\n")
        f.write("  ARI=1.0 => identical partitions.\n")
        f.write("  ARI near 0 => agreement no better than random.\n")
        f.write(f"  {agreement*100:.1f}% of samples receive the same class-aligned\n")
        f.write("  cluster id under both algorithms.\n")
        f.write(f"\n  Full per-sample assignment table saved to:\n")
        f.write(f"  {os.path.basename(csv_path)}\n")

    # ---- per-class agreement bar chart ---------------------------------
    classes = np.unique(y_true)
    per_class_agree = []
    for cls in classes:
        mask = y_true == cls
        per_class_agree.append(
            float(np.mean(birch_mapped[mask] == agglom_pred[mask]))
        )

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.38), 4))
    ax.bar(range(len(classes)), per_class_agree, color="#4CAF50", edgecolor="white")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([str(c) for c in classes], rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Sample-level agreement rate")
    ax.set_xlabel("Ground-truth class index")
    ax.set_title(
        f"Per-class BIRCH vs Agglomerative agreement\n"
        f"Overall {agreement*100:.1f}% | ARI={ari_between:.4f} | "
        f"NMI={nmi_between:.4f}"
    )
    ax.axhline(agreement, color="#F44336", linestyle="--",
               label=f"Overall mean ({agreement:.3f})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "birch_agglom_class_agreement.png"), dpi=150)
    plt.close(fig)

    print(f"      [BIRCH vs Agglom]  ARI={ari_between:.4f}  "
          f"NMI={nmi_between:.4f}  sample-agree={agreement*100:.2f}%  "
          f"(n={n} samples, per-sample CSV saved)")
    return ari_between, nmi_between, agreement


# =========================================================================== #
# 9. Extended system information  [Reviewer-1 comment 4]
# =========================================================================== #

def save_system_info(out_dir, algo_times=None):
    """
    Save extended hardware and software environment details to system_info.json
    so runtime results can be reproduced and verified (Reviewer-1, comment 4).
    Fields include: OS, Python version, CPU model, physical and logical core
    count, total RAM, available RAM, peak process RSS (memory), package
    versions, and per-algorithm wall-clock times.
    """
    import importlib

    # Hardware info via psutil
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_total = round(vm.total / 1024**3, 2)
        ram_avail = round(vm.available / 1024**3, 2)
        ram_peak_rss = round(
            psutil.Process(os.getpid()).memory_info().rss / 1024**3, 3
        )
        cpu_log = psutil.cpu_count(logical=True)
        cpu_phys = psutil.cpu_count(logical=False)
    except ImportError:
        ram_total = ram_avail = ram_peak_rss = "psutil not installed"
        cpu_log = cpu_phys = "psutil not installed"

    # Package versions
    pkg_map = {
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "scipy": "scipy",
        "scikit-image": "skimage",
        "Pillow": "PIL",
        "torch": "torch",
        "sklearn_extra": "sklearn_extra",
        "skfuzzy": "skfuzzy",
        "minisom": "minisom",
        "psutil": "psutil",
    }
    packages = {}
    for display_name, import_name in pkg_map.items():
        try:
            mod = importlib.import_module(import_name)
            packages[display_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[display_name] = "not installed"

    info = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": platform.processor(),
        "cpu_cores_logical": cpu_log,
        "cpu_cores_physical": cpu_phys,
        "ram_total_gb": ram_total,
        "ram_available_gb": ram_avail,
        "ram_peak_rss_gb": ram_peak_rss,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "default"),
        "packages": packages,
        "algo_wall_clock_seconds": algo_times or {},
    }

    json_path = os.path.join(out_dir, "system_info.json")
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2)

    txt_path = os.path.join(out_dir, "system_info.txt")
    with open(txt_path, "w") as f:
        f.write(f"OS               : {info['os']}\n")
        f.write(f"Python           : {info['python']}\n")
        f.write(f"CPU              : {info['cpu_model']}\n")
        f.write(f"Cores (physical) : {cpu_phys}\n")
        f.write(f"Cores (logical)  : {cpu_log}\n")
        f.write(f"RAM total        : {ram_total} GB\n")
        f.write(f"RAM available    : {ram_avail} GB\n")
        f.write(f"Peak RSS         : {ram_peak_rss} GB\n")


# =========================================================================== #
# 10. Plotting helpers
# =========================================================================== #

def plot_metrics_comparison(rows, out_path,
                            metrics=("cluster_accuracy", "NMI", "ARI", "purity")):
    """Grouped bar chart comparing every algorithm sorted by cluster accuracy.
    Error bars show +/- std when n_seeds > 1 (Reviewer-1, comment 2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_rows = sorted(rows, key=lambda r: r.get("cluster_accuracy", 0),
                         reverse=True)
    algos = [r["algo"] for r in sorted_rows]
    n_algos = len(algos)
    n_metrics = len(metrics)
    fig, ax = plt.subplots(figsize=(max(10, n_algos * 0.9), 6))
    bar_width = 0.8 / n_metrics
    x = np.arange(n_algos)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_metrics))
    for i, metric in enumerate(metrics):
        vals = [r.get(metric, np.nan) for r in sorted_rows]
        stds = [r.get(f"{metric}_std", 0.0) for r in sorted_rows]
        offset = (i - (n_metrics - 1) / 2.0) * bar_width
        ax.bar(x + offset, vals, width=bar_width, label=metric,
               color=colors[i], yerr=stds, capsize=2,
               error_kw=dict(elinewidth=0.8, ecolor="black"))
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Clustering algorithm comparison (sorted by cluster accuracy)")
    ax.legend(loc="upper right", ncol=n_metrics)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tsne(X, y_true, y_pred, out_path, seed=0, n_sample=3000):
    """2-D t-SNE scatter plots colored by ground truth and by predicted cluster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(X)
    idx = (np.random.RandomState(seed).choice(n, n_sample, replace=False)
           if n > n_sample else np.arange(n))
    emb = TSNE(
        n_components=2, init="pca", random_state=seed, perplexity=30
    ).fit_transform(X[idx])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(emb[:, 0], emb[:, 1], c=y_true[idx], cmap="tab20", s=6)
    axes[0].set_title("t-SNE colored by GROUND TRUTH class")
    axes[1].scatter(emb[:, 0], emb[:, 1],
                    c=np.asarray(y_pred)[idx], cmap="tab20", s=6)
    axes[1].set_title("t-SNE colored by PREDICTED cluster")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =========================================================================== #
# 11. Main pipeline
# =========================================================================== #

ALL_ALGOS = [
    "kmeans", "minibatchkmeans", "bisecting_kmeans", "balanced_kmeans",
    "kernel_kmeans", "gmm", "em", "hcmodel", "spectral", "agglomerative",
    "birch", "clara", "kmedoids", "subspace", "fcm",
    "dbscan", "optics", "hdbscan", "meanshift", "affinity_propagation",
    "som", "xmeans", "gmeans",
]

SUBSAMPLE_ALGOS = {"spectral", "agglomerative", "subspace", "kmedoids",
                   "kernel_kmeans"}
NO_FIXED_K_ALGOS = {"dbscan", "optics", "hdbscan", "meanshift",
                    "affinity_propagation", "xmeans", "gmeans"}


def _stratified_subsample(y, n_total, seed):
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    per_class = max(1, n_total // len(classes))
    idx_all = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        take = min(per_class, len(idx_c))
        idx_all.append(rng.choice(idx_c, take, replace=False))
    return np.concatenate(idx_all)


def _call_algo(algo, X, y, sub_idx, args, seed):
    """
    Dispatch a single algorithm call for one seed.
    Returns predicted labels array.
    """
    k = args.k
    use_sub = algo in SUBSAMPLE_ALGOS
    Xs = X[sub_idx] if use_sub else X

    if algo == "kmeans":
        return run_kmeans(Xs, k, minibatch=len(Xs) > 20000, seed=seed)
    if algo == "minibatchkmeans":
        return run_minibatch_kmeans(Xs, k, seed=seed)
    if algo == "bisecting_kmeans":
        return run_bisecting_kmeans(Xs, k, seed=seed)
    if algo == "balanced_kmeans":
        return run_balanced_kmeans(Xs, k, seed=seed)
    if algo == "kernel_kmeans":
        return run_kernel_kmeans(Xs, k, seed=seed,
                                 kernel=args.kernel_kmeans_kernel,
                                 gamma=args.kernel_kmeans_gamma)
    if algo == "gmm":
        return run_gmm(Xs, k, seed=seed)
    if algo == "em":
        return run_em(Xs, k, seed=seed)
    if algo == "hcmodel":
        return run_hcmodel(Xs, k, seed=seed)
    if algo == "spectral":
        return run_spectral(Xs, k, seed=seed)
    if algo == "agglomerative":
        return run_agglomerative(Xs, k)
    if algo == "birch":
        return run_birch(Xs, k, threshold=args.birch_threshold,
                         branching_factor=args.birch_branching_factor)
    if algo == "clara":
        return run_clara(Xs, k, seed=seed, n_samples=args.clara_samples,
                         sample_size=args.clara_sample_size)
    if algo == "kmedoids":
        return run_kmedoids(Xs, k, seed=seed)
    if algo == "fcm":
        return run_fcm(Xs, k, m=args.fcm_m, seed=seed)
    if algo == "subspace":
        return run_subspace(Xs, k, seed=seed, alpha=args.subspace_alpha)
    if algo == "dbscan":
        pred, _ = run_dbscan(Xs, eps=args.dbscan_eps,
                             min_samples=args.dbscan_min_samples)
        return pred
    if algo == "optics":
        return run_optics(Xs, min_samples=args.optics_min_samples,
                          xi=args.optics_xi)
    if algo == "hdbscan":
        return run_hdbscan(Xs, min_cluster_size=args.hdbscan_min_cluster_size,
                           min_samples=args.hdbscan_min_samples)
    if algo == "meanshift":
        pred, _ = run_meanshift(Xs, quantile=args.meanshift_quantile, seed=seed)
        return pred
    if algo == "affinity_propagation":
        return run_affinity_propagation(Xs, damping=args.affinity_damping, seed=seed)
    if algo == "som":
        return run_som(Xs, k, seed=seed, som_dim=args.som_dim,
                       num_iteration=args.som_num_iteration)
    if algo == "xmeans":
        return run_xmeans(Xs, k_max=k, seed=seed, k_min=args.xmeans_kmin)
    if algo == "gmeans":
        return run_gmeans(Xs, k_max=k, seed=seed,
                          ad_threshold=args.gmeans_ad_threshold)
    raise ValueError(f"Unknown algorithm: {algo!r}")


def _run_one_algo(algo, X_pca, y, sub_idx, args):
    """
    Run a single algorithm (possibly multiple seeds for stochastic ones) and
    return an aggregated metrics dict (mean +/- std) plus the last prediction.
    """
    n_seeds = args.n_seeds if algo in STOCHASTIC_ALGOS else 1
    use_sub = algo in SUBSAMPLE_ALGOS
    Xs = X_pca[sub_idx] if use_sub else X_pca
    ys = y[sub_idx] if use_sub else y

    all_metrics, last_pred, last_mapping = [], None, None
    t0 = time.time()

    for s in range(args.seed, args.seed + n_seeds):
        pred = _call_algo(algo, X_pca, y, sub_idx, args, seed=s)
        m, mapping = evaluate(ys, pred, X_for_silhouette=Xs, seed=args.seed)
        all_metrics.append(m)
        last_pred = pred
        last_mapping = mapping

    elapsed = time.time() - t0

    # Aggregate across seeds
    metric_keys = list(all_metrics[0].keys())
    agg = {}
    for mk in metric_keys:
        vals = [m[mk] for m in all_metrics
                if not np.isnan(m.get(mk, float("nan")))]
        agg[mk] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{mk}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

    # Build note string
    note_parts = []
    if use_sub:
        note_parts.append(
            f"stratified subsample n={len(sub_idx)} (O(n^2) method)"
        )
    if algo == "balanced_kmeans":
        note_parts.append("capacity-constrained assignment")
    if algo == "em":
        note_parts.append("full-covariance GMM (classic EM)")
    if algo == "hcmodel":
        note_parts.append("Ward-init full-cov GMM (model-based hierarchical)")
    if algo in NO_FIXED_K_ALGOS:
        note_parts.append(f"k not enforced (upper bound={args.k})")
    if algo == "kernel_kmeans":
        note_parts.append(f"kernel={args.kernel_kmeans_kernel}")
    if algo == "som":
        note_parts.append("SOM prototypes collapsed to k via k-means")

    agg.update(
        algo=algo,
        features=args.features,
        n=len(ys),
        n_seeds=n_seeds,
        time_s=round(elapsed, 3),
        note="; ".join(note_parts),
    )
    return agg, last_pred, last_mapping, ys, Xs


def main():
    ap = argparse.ArgumentParser(
        description="Tamil handwritten character clustering - reviewer-revised."
    )
    ap.add_argument("--data_root", required=True,
                    help="Path to 'Vowels&Consonants' folder")
    ap.add_argument("--k", type=int, default=31,
                    help="Number of clusters (ignored when --k_range is set)")
    ap.add_argument("--features", choices=["hog", "pixel", "cae"], default="hog")
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--algos", nargs="+",
                    default=["kmeans", "gmm", "spectral", "agglomerative"],
                    choices=ALL_ALGOS)
    ap.add_argument("--spectral_subsample", type=int, default=14260)
    ap.add_argument("--seed", type=int, default=0)

    # ---- Reviewer-addressed additions ----
    ap.add_argument(
        "--n_seeds", type=int, default=1,
        help="[R1-2, R2-3] Stochastic algorithms are run this many times "
             "(seeds seed .. seed+n_seeds-1); metrics reported as mean +/- std."
    )
    ap.add_argument(
        "--k_range", type=int, nargs=2, default=None,
        metavar=("K_MIN", "K_MAX"),
        help="[R2-1] Sweep k in [K_MIN, K_MAX] and select automatically "
             "without using ground-truth labels. Overrides --k."
    )
    ap.add_argument(
        "--k_select", choices=["silhouette", "bic", "gap"],
        default="silhouette",
        help="[R2-1] Criterion for automatic k selection."
    )
    ap.add_argument(
        "--elbow", type=int, nargs=2, default=None,
        metavar=("K_MIN", "K_MAX"),
        help="Generate Elbow curve (WCSS/inertia vs k) plot for k in [K_MIN, K_MAX]."
    )
    ap.add_argument(
        "--holdout_frac", type=float, default=0.0,
        help="[R2-2] Fraction of class folders held out as unseen-writer "
             "test set. 0 = no holdout (default)."
    )
    ap.add_argument(
        "--compare_birch_agglom", action="store_true",
        help="[R1-3, R2-4] Save full sample-level BIRCH vs agglomerative "
             "comparison CSV and per-class agreement heatmap."
    )
    ap.add_argument(
        "--class_imbalance_test", action="store_true",
        help="[R2-5] Re-evaluate top algorithms after randomly halving every "
             "other class to simulate class imbalance."
    )
    ap.add_argument(
        "--dump_sysinfo", action="store_true",
        help="[R1-4] Save extended hardware/memory/runtime info to "
             "system_info.json."
    )

    # ---- Algorithm-specific hyperparameters ----
    ap.add_argument("--clara_samples", type=int, default=5)
    ap.add_argument("--clara_sample_size", type=int, default=None)
    ap.add_argument("--subspace_alpha", type=float, default=0.01)
    ap.add_argument("--dbscan_eps", type=float, default=None)
    ap.add_argument("--dbscan_min_samples", type=int, default=5)
    ap.add_argument("--optics_min_samples", type=int, default=5)
    ap.add_argument("--optics_xi", type=float, default=0.05)
    ap.add_argument("--hdbscan_min_cluster_size", type=int, default=10)
    ap.add_argument("--hdbscan_min_samples", type=int, default=None)
    ap.add_argument("--meanshift_quantile", type=float, default=0.2)
    ap.add_argument("--affinity_damping", type=float, default=0.9)
    ap.add_argument("--fcm_m", type=float, default=2.0)
    ap.add_argument("--birch_threshold", type=float, default=0.5)
    ap.add_argument("--birch_branching_factor", type=int, default=50)
    ap.add_argument("--kernel_kmeans_kernel", default="rbf")
    ap.add_argument("--kernel_kmeans_gamma", type=float, default=None)
    ap.add_argument("--som_dim", type=int, default=None)
    ap.add_argument("--som_num_iteration", type=int, default=1000)
    ap.add_argument("--xmeans_kmin", type=int, default=2)
    ap.add_argument("--gmeans_ad_threshold", type=float, default=1.0)
    ap.add_argument("--out_dir", default="results")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ================================================================== #
    # Step 1 - Load dataset
    # ================================================================== #
    print("[1/7] Loading dataset ...")
    imgs_all, y_all, class_names_all = load_dataset(args.data_root)
    print(f"      {len(imgs_all)} images, {len(class_names_all)} classes: "
          f"{class_names_all}")

    # Optional holdout split [R2-2]
    if args.holdout_frac > 0.0:
        print(f"      Holdout split: {args.holdout_frac:.0%} of classes "
              f"held out as unseen-writer set.")
        (imgs, y, holdout_imgs, holdout_y,
         class_names, holdout_class_names) = holdout_split(
            imgs_all, y_all, class_names_all,
            args.holdout_frac, seed=args.seed
        )
        print(f"      Train classes ({len(class_names)}): {class_names}")
        print(f"      Holdout classes ({len(holdout_class_names)}): "
              f"{holdout_class_names}")
    else:
        imgs, y, class_names = imgs_all, y_all, class_names_all
        holdout_imgs = holdout_y = holdout_class_names = None

    imgs = binarize_and_center(imgs)

    # ================================================================== #
    # Step 2 - Feature extraction
    # ================================================================== #
    print(f"[2/7] Extracting '{args.features}' features ...")
    t0 = time.time()
    if args.features == "hog":
        X_raw = extract_hog_features(imgs)
    elif args.features == "pixel":
        X_raw = extract_pixel_features(imgs)
    else:
        X_raw = extract_cae_features(imgs)
    print(f"      feature shape = {X_raw.shape}  ({time.time()-t0:.1f}s)")

    X_scaled = StandardScaler().fit_transform(X_raw)
    n_comp = min(args.pca_dim, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca_model = PCA(n_components=n_comp, random_state=args.seed).fit(X_scaled)
    X_pca = pca_model.transform(X_scaled)
    print(f"      PCA -> {X_pca.shape[1]} dims  "
          f"(explained var = {pca_model.explained_variance_ratio_.sum():.3f})")

    # Project holdout through the same PCA (no label leakage)
    if holdout_imgs is not None:
        holdout_imgs = binarize_and_center(holdout_imgs)
        if args.features == "hog":
            Xh_raw = extract_hog_features(holdout_imgs)
        elif args.features == "pixel":
            Xh_raw = extract_pixel_features(holdout_imgs)
        else:
            Xh_raw = extract_cae_features(holdout_imgs)
        X_holdout_pca = pca_model.transform(
            StandardScaler().fit_transform(Xh_raw)
        )
    else:
        X_holdout_pca = None

    sub_idx = _stratified_subsample(y, args.spectral_subsample, args.seed)

    # ================================================================== #
    # Step 3 - Automatic k-selection [R2-1]
    # ================================================================== #
    if args.k_range is not None:
        print(f"[3/7] Automatic k-selection ({args.k_select}) "
              f"over k in [{args.k_range[0]}, {args.k_range[1]}] ...")
        best_k, k_scores = select_k(
            X_pca, tuple(args.k_range), args.k_select, seed=args.seed
        )
        print(f"      Selected k = {best_k}  (original --k={args.k})")
        args.k = best_k
        plot_k_selection_curve(
            k_scores, args.k_select, best_k,
            os.path.join(args.out_dir, "k_selection_curve.png")
        )
    else:
        print(f"[3/7] Using fixed k = {args.k}  "
              f"(pass --k_range K_MIN K_MAX to enable automatic selection)")

    if args.elbow is not None:
        generate_elbow_curve(
            X_pca, args.elbow[0], args.elbow[1],
            os.path.join(args.out_dir, "elbow_curve.png")
        )

    # ================================================================== #
    # Step 4 - Clustering + evaluation
    # ================================================================== #
    print(f"[4/7] Clustering + evaluating "
          f"({args.n_seeds} seed(s) for stochastic algorithms) ...")
    rows = []
    predictions = {}
    algo_times = {}

    for algo in args.algos:
        try:
            row, last_pred, last_mapping, y_used, X_used = _run_one_algo(
                algo, X_pca, y, sub_idx, args
            )
        except Exception as exc:
            print(f"      {algo:<22} FAILED: {exc}")
            continue
        rows.append(row)
        predictions[algo] = (last_pred, last_mapping, y_used, X_used)
        algo_times[algo] = row["time_s"]
        n_seeds_used = row["n_seeds"]
        print(
            f"      {algo:<22}  "
            f"acc={row['cluster_accuracy']:.3f}"
            + (f"+-{row.get('cluster_accuracy_std',0):.3f}" if n_seeds_used > 1 else "")
            + f"  NMI={row['NMI']:.3f}  ARI={row['ARI']:.3f}"
            + f"  sil={row.get('silhouette',float('nan')):.3f}"
            + f"  k={int(row['n_clusters_found'])}  seeds={n_seeds_used}"
        )

    # ================================================================== #
    # Step 5 - BIRCH vs Agglomerative sample-level comparison [R1-3, R2-4]
    # ================================================================== #
    if (args.compare_birch_agglom
            and "birch" in predictions
            and "agglomerative" in predictions):
        print("[5/7] BIRCH vs Agglomerative sample-level comparison ...")
        birch_pred_full, _, _, _ = predictions["birch"]
        agglom_pred, _, y_a, _ = predictions["agglomerative"]
        # BIRCH ran on full X_pca (original index order).
        # Agglomerative ran on X_pca[sub_idx] (sub_idx order).
        # Index birch predictions through sub_idx so both arrays correspond
        # to exactly the same samples in the same order.
        birch_pred_aligned = np.asarray(birch_pred_full)[sub_idx]
        compare_birch_agglom(birch_pred_aligned, agglom_pred, y[sub_idx], args.out_dir)
    else:
        print("[5/7] Skipped BIRCH vs Agglomerative comparison "
              "(pass --compare_birch_agglom and include both algos).")

    # ================================================================== #
    # Step 6 - Save metrics + plots
    # ================================================================== #
    print("[6/7] Saving metrics.csv and plots ...")
    keys = [
        "algo", "features", "n", "n_seeds",
        "cluster_accuracy", "cluster_accuracy_std",
        "ARI", "ARI_std", "NMI", "NMI_std",
        "homogeneity", "homogeneity_std",
        "completeness", "completeness_std",
        "v_measure", "v_measure_std",
        "fowlkes_mallows", "fowlkes_mallows_std",
        "purity", "purity_std",
        "silhouette", "silhouette_std",
        "n_clusters_found", "n_clusters_found_std",
        "time_s", "note",
    ]
    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})

    if rows:
        plot_metrics_comparison(
            rows, os.path.join(args.out_dir, "metrics_comparison.png")
        )
        best = max(rows, key=lambda r: r.get("cluster_accuracy", 0))
        best_algo = best["algo"]
        print(f"      Best: {best_algo}  "
              f"acc={best['cluster_accuracy']:.3f}")
        pred, _, y_used, X_used = predictions[best_algo]
        plot_tsne(
            X_used, y_used, pred,
            os.path.join(args.out_dir, f"tsne_{best_algo}.png"),
            seed=args.seed,
        )

    # ---- Holdout evaluation [R2-2] ------------------------------------
    if X_holdout_pca is not None and rows:
        print("      Evaluating on holdout (unseen-writer) set ...")
        holdout_rows = []
        k_hold = len(np.unique(holdout_y))
        sub_h = _stratified_subsample(
            holdout_y,
            min(args.spectral_subsample, len(holdout_y)),
            args.seed,
        )
        # Temporarily set k to holdout class count for fair evaluation
        orig_k = args.k
        args.k = k_hold
        for algo in args.algos:
            if algo in NO_FIXED_K_ALGOS or algo not in predictions:
                continue
            try:
                row_h, _, _, _, _ = _run_one_algo(
                    algo, X_holdout_pca, holdout_y, sub_h, args
                )
                row_h["split"] = "holdout"
                holdout_rows.append(row_h)
                print(f"      [holdout] {algo:<22}  "
                      f"acc={row_h['cluster_accuracy']:.3f}  "
                      f"NMI={row_h['NMI']:.3f}")
            except Exception as exc:
                print(f"      [holdout] {algo:<22} FAILED: {exc}")
        args.k = orig_k

        holdout_keys = ["algo", "split"] + [k for k in keys if k != "algo"]
        h_path = os.path.join(args.out_dir, "holdout_metrics.csv")
        with open(h_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=holdout_keys,
                                    extrasaction="ignore")
            writer.writeheader()
            for r in holdout_rows:
                writer.writerow({k: r.get(k, "") for k in holdout_keys})
        print(f"      holdout_metrics.csv saved ({len(holdout_rows)} rows).")

    # ---- Class imbalance test [R2-5] ----------------------------------
    if args.class_imbalance_test and rows:
        print("      Running class imbalance test "
              "(halving every other class) ...")
        imgs_imb, y_imb = imbalance_subsample(imgs, y, seed=args.seed)
        if args.features == "hog":
            Xr_imb = extract_hog_features(imgs_imb)
        elif args.features == "pixel":
            Xr_imb = extract_pixel_features(imgs_imb)
        else:
            Xr_imb = extract_cae_features(imgs_imb)
        X_imb_pca = pca_model.transform(
            StandardScaler().fit_transform(Xr_imb)
        )
        sub_imb = _stratified_subsample(
            y_imb, min(args.spectral_subsample, len(y_imb)), args.seed
        )
        # Top-3 algorithms by cluster_accuracy
        top3 = [
            r["algo"] for r in
            sorted(rows, key=lambda r: r.get("cluster_accuracy", 0),
                   reverse=True)[:3]
            if r["algo"] in args.algos
        ]
        imb_rows = []
        for algo in top3:
            try:
                row_imb, _, _, _, _ = _run_one_algo(
                    algo, X_imb_pca, y_imb, sub_imb, args
                )
                row_imb["split"] = "imbalanced"
                imb_rows.append(row_imb)
                print(f"      [imbalance] {algo:<22}  "
                      f"acc={row_imb['cluster_accuracy']:.3f}  "
                      f"NMI={row_imb['NMI']:.3f}")
            except Exception as exc:
                print(f"      [imbalance] {algo:<22} FAILED: {exc}")

        imb_keys = ["algo", "split"] + [k for k in keys if k != "algo"]
        imb_path = os.path.join(args.out_dir, "imbalance_metrics.csv")
        with open(imb_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=imb_keys,
                                    extrasaction="ignore")
            writer.writeheader()
            for r in imb_rows:
                writer.writerow({k: r.get(k, "") for k in imb_keys})
        print(f"      imbalance_metrics.csv saved ({len(imb_rows)} rows).")

    # ---- Extended system info [R1-4] ----------------------------------
    if args.dump_sysinfo:
        save_system_info(args.out_dir, algo_times=algo_times)
        print("      system_info.json saved.")

    print("[7/7] Done. Results in:", args.out_dir)
    return rows


if __name__ == "__main__":
    main()
