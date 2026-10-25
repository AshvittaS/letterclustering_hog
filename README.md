# Tamil Handwritten Character Clustering

An unsupervised machine learning project for clustering offline Tamil handwritten vowels and consonants using **Histogram of Oriented Gradients (HOG), Principal Component Analysis (PCA), and classical clustering algorithms**.

The project explores how effectively handwritten Tamil characters can be grouped by visual similarity without using class labels during training. Ground-truth labels are used only after clustering to evaluate the results.

A Gradio interface allows users to draw or upload a Tamil character and explore its closest learned clusters.

Demo:https://huggingface.co/spaces/Ashvitta07/letterclustering_hog

<img width="1723" height="925" alt="image" src="https://github.com/user-attachments/assets/f0d55a51-bc48-4627-a2ca-d8853115011d" />
<img width="1892" height="878" alt="image" src="https://github.com/user-attachments/assets/7e5c4b88-c54f-4743-9834-6b6b0984477c" />


## Project Overview

- **Learning approach:** Unsupervised learning
- **Feature extraction:** Histogram of Oriented Gradients (HOG)
- **Dimensionality reduction:** PCA
- **Best-performing algorithm:** Spectral Clustering
- **Number of clusters:** 31
- **Dataset size:** 14,260 handwritten glyphs
- **Input image size:** 64 × 64 pixels
- **Interface:** Gradio

## Dataset

The project uses a 31-class subset of the HP Tamil Handwritten Character Dataset.

| Property | Description |
|---|---|
| Character classes | 31 |
| Vowels | 12 |
| Consonants | 18 |
| Ayudha letter | 1 |
| Samples per class | 460 |
| Total samples | 14,260 |
| Image dimensions | 64 × 64 pixels |
| Image representation | Binary bitmap |
| Class distribution | Balanced |

The dataset contains handwritten variations of Tamil characters, making it suitable for studying visual similarity and unsupervised character grouping.

**Important:** Class labels are excluded from feature learning and clustering. They are used only during post-clustering evaluation.

## Clustering Pipeline

The pipeline transforms handwritten glyphs into feature vectors, reduces their dimensionality, and groups visually similar samples.

### 1. Image Preprocessing

- Converts glyph images to grayscale where necessary.
- Applies binarization to represent pixels using binary values.
- Standardizes the input representation to 64 × 64 pixels.

### 2. HOG Feature Extraction

Histogram of Oriented Gradients captures local edge directions and structural patterns in handwritten characters.

Configuration:
- Orientations: 9
- Pixels per cell: 8 × 8
- Cells per block: 2 × 2
- Block normalization: L2-Hys
- Feature dimensionality: 1,764

These features represent the shape and stroke structure of each glyph.

### 3. Feature Scaling and PCA

`StandardScaler` standardizes the extracted features before PCA reduces their dimensionality.

- Original feature dimensions: 1,764
- PCA components: 100
- Explained variance: 80.3%

This reduces the computational cost of clustering while preserving much of the variation in the HOG features.

### 4. Unsupervised Clustering

The project compares seven clustering approaches using 31 clusters.

| Algorithm | Purpose |
|---|---|
| K-Means | Groups samples around cluster centroids |
| MiniBatch K-Means | Uses mini-batches for faster centroid optimization |
| Balanced K-Means | Explores clustering with balanced cluster assignments |
| Gaussian Mixture Model (EM) | Models clusters using probability distributions |
| BIRCH | Builds a clustering-feature hierarchy for scalable clustering |
| Spectral Clustering | Uses a similarity graph to identify clusters |
| Agglomerative Clustering (Ward) | Builds a hierarchy of clusters by merging groups |

The algorithms are compared under the same general HOG and PCA feature representation.

## Experimental Results

The following results were obtained from a single experimental configuration with 14,260 samples, 31 clusters, and 100 PCA components, except where the stability experiment is explicitly noted.

| Algorithm | Matched Accuracy | ARI | NMI | Purity | Silhouette |
|---|---:|---:|---:|---:|---:|
| K-Means | 0.817 | 0.704 | 0.817 | 0.832 | 0.106 |
| MiniBatch K-Means | 0.758 | 0.683 | 0.814 | 0.784 | 0.107 |
| Balanced K-Means | 0.814 | 0.694 | 0.788 | 0.817 | 0.103 |
| Gaussian Mixture Model (EM) | 0.785 | 0.693 | 0.831 | 0.800 | 0.102 |
| BIRCH | 0.841 | 0.745 | 0.868 | 0.848 | 0.102 |
| **Spectral Clustering** | **0.870** | **0.744** | **0.883** | **0.876** | 0.104 |
| Agglomerative (Ward) | 0.841 | 0.745 | 0.868 | 0.848 | 0.102 |

### Key Findings

- Spectral Clustering achieves the highest reported matched accuracy of **87.0%**.
- It also obtains the highest NMI of **0.883** and purity of **0.876**.
- Across eight random seeds, the reported Spectral Clustering accuracy is **0.8696 ± 0.0003**.
- BIRCH and Agglomerative Clustering achieve competitive results, with a matched accuracy of 84.1%.
- The silhouette scores are relatively low across the experiments, indicating that separation between clusters in the evaluated feature space is not especially strong.

### Evaluation Metrics

- **Hungarian-matched accuracy:** Measures the agreement between predicted clusters and ground-truth classes after finding the optimal one-to-one cluster-to-class assignment.
- **Adjusted Rand Index (ARI):** Measures agreement between two groupings while adjusting for chance.
- **Normalized Mutual Information (NMI):** Measures the shared information between predicted clusters and true classes.
- **Purity:** Measures the proportion of samples assigned to the dominant class within each cluster.
- **Silhouette score:** Measures how well samples fit within their own clusters compared with neighboring clusters.
- **Confusion matrix:** Visualizes the relationship between ground-truth classes and predicted clusters.

Because clustering IDs are arbitrary, raw cluster numbers are not directly comparable to character labels. The Hungarian matching is used for the reported matched accuracy.

## Gradio Interface

The interface provides two main tabs for exploring the trained clustering model.

### 1. Character Clustering

Users can draw a Tamil character or upload a handwritten glyph.

The interface provides:

- **Input preview:** Displays the 64 × 64 image used by the model.
- **Closest clusters:** Displays the three nearest clusters for the input.
- **Confidence scores:** Shows the application's cluster-related confidence or similarity estimates.
- **Example glyphs:** Displays representative handwritten samples from the closest cluster.

The input image is processed using the same preprocessing, scaler, and PCA transformation used during training. Its transformed representation is compared with stored training representations to identify the closest learned groups.

**Interpretation:** Cluster membership indicates visual similarity to the learned groups; it does not guarantee that the input character has been correctly recognized.

### 2. Cluster Explorer and Evaluation

The second tab provides tools for inspecting the learned clustering structure.

It includes:

- Evaluation metrics for the clustering experiment.
- A t-SNE visualization of the dataset and predicted clusters.
- A cluster browser for exploring groups of handwritten glyphs.
- Visual comparisons between ground-truth classes and predicted clusters.

These visualizations help identify well-separated groups, overlapping characters, and handwritten forms that may be difficult to distinguish using HOG features alone.

## Visual Analysis

### t-SNE Visualization

The t-SNE visualization compares the dataset's ground-truth classes with the clusters predicted by Spectral Clustering.

The plots help reveal:
- Whether samples belonging to the same character form compact groups.
- Whether visually similar characters occupy neighboring regions.
- Which character groups overlap and may require better feature representations.

### Confusion Matrix

The confusion matrix summarizes the correspondence between the 31 ground-truth classes and the predicted Spectral Clustering groups.

A strong diagonal after appropriate cluster-to-class matching indicates good correspondence. Off-diagonal entries highlight classes that are confused or share similar visual structures.

## Repository Structure

```text
.
├── app.py
├── cluster.py
├── export_model.py
├── model_bundle.joblib
├── model_bundle_tsne.png
├── requirements.txt
└── assets/
    ├── demo.png
    ├── tsne_spectral.png
    └── confusion_matrix.png
```

| File | Description |
|---|---|
| `app.py` | Implements the Gradio interface for character input, cluster exploration, and inference using the saved model bundle. |
| `cluster.py` | Contains feature extraction, clustering algorithms, evaluation metrics, and experimental analysis. |
| `export_model.py` | Trains the selected clustering pipeline and exports the fitted artifacts into a reusable model bundle. |
| `model_bundle.joblib` | Stores the exported preprocessing and dimensionality-reduction objects, clustering results, and associated inference artifacts, depending on the export implementation. |
| `model_bundle_tsne.png` | Provides the t-SNE visualization used by the interface. |
| `requirements.txt` | Lists the Python dependencies required by the project. |
| `assets/demo.png` | Contains the interface demonstration image. |
| `assets/tsne_spectral.png` | Contains the Spectral Clustering t-SNE comparison. |
| `assets/confusion_matrix.png` | Contains the Spectral Clustering confusion matrix. |

## Scope and Limitations

- The experiment focuses on a balanced subset of 31 Tamil handwritten character classes rather than the entire Tamil script.
- HOG primarily captures local edge and stroke information; more advanced learned representations may capture additional structural variation.
- The number of clusters is fixed at 31, matching the number of evaluated character classes. This is a chosen experimental setting, not a requirement of unsupervised clustering.
- The reported accuracy depends on post-hoc Hungarian matching and should not be interpreted as ordinary supervised classification accuracy.
- Cluster confidence scores in the interface should be interpreted according to their actual implementation; they are not necessarily calibrated probabilities.
- Results may differ with preprocessing choices, feature representations, clustering parameters, and dataset composition.

## Future Improvements

- Compare HOG features with CNN embeddings and self-supervised visual representations.
- Evaluate clustering quality across different PCA dimensions and cluster counts.
- Investigate methods for identifying visually similar Tamil character pairs.
- Explore density-based clustering and alternative similarity metrics.
- Improve confidence estimation for new handwritten inputs.
- Extend the experiments to additional Tamil characters and more diverse handwriting samples.

## Summary

This project demonstrates an end-to-end unsupervised learning pipeline for Tamil handwritten character clustering. By combining HOG feature extraction, PCA, and seven classical clustering algorithms, it evaluates how well handwritten glyphs can be grouped without using labels during training.

**Spectral Clustering achieves the best reported matched accuracy of 87.0%**, while the Gradio interface makes the learned groups accessible through interactive character input, cluster comparisons, and visual evaluation tools.
