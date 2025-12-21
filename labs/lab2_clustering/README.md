# Lab Work #2: Clustering and Decision Trees

## Objective
Learn clustering and classification methods using Python (as an alternative to WEKA):
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN Clustering
- Decision Tree Classification

## Theoretical Background

### Clustering Methods

#### K-Means
Partitions data into K clusters by minimizing within-cluster variance:
```
min Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
```

#### Hierarchical Clustering
Builds a tree of clusters using linkage methods:
- **Ward**: Minimizes variance increase
- **Complete**: Maximum pairwise distance
- **Average**: Mean pairwise distance
- **Single**: Minimum pairwise distance

#### DBSCAN
Density-based clustering that can identify noise points:
- **eps**: Maximum distance between neighbors
- **min_samples**: Minimum points to form a cluster

### Decision Trees
Classification trees that split data based on feature thresholds:
- **Gini impurity** or **Entropy** for split selection
- **Pruning** to prevent overfitting

### Evaluation Metrics
- **Silhouette Score**: Cluster cohesion vs separation (-1 to 1)
- **Calinski-Harabasz**: Ratio of between/within cluster dispersion
- **Accuracy, Precision, Recall, F1-Score** for classification

## Implementation

### Files
- `clustering_analysis.py` - Main implementation

### Usage
```bash
source venv/bin/activate
cd labs/lab2_clustering
python clustering_analysis.py
```

### Example: BMW Customer Classification
Predict which customers will buy BMW M5 based on:
- Age
- Income level
- Number of cars owned
- Family status
- Children
- Home ownership

## Tasks Completed

1. ✅ K-Means clustering with elbow method
2. ✅ Silhouette analysis for optimal k
3. ✅ Hierarchical clustering with dendrogram
4. ✅ DBSCAN for density-based clustering
5. ✅ Decision tree classification
6. ✅ Feature importance analysis
7. ✅ Confusion matrix visualization
8. ✅ Cross-validation evaluation

## Output Files
- `elbow_silhouette.png` - Optimal k analysis
- `kmeans_clusters.png` - K-Means results
- `dendrogram.png` - Hierarchical clustering dendrogram
- `decision_tree.png` - Tree visualization
- `confusion_matrix.png` - Classification results
- `feature_importance.png` - Feature rankings

## Dependencies
- numpy, pandas
- scikit-learn
- matplotlib, seaborn
- scipy
