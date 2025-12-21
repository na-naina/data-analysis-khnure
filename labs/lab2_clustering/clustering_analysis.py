"""
Lab Work #2: Clustering and Decision Trees
===========================================
Clustering Analysis and Classification using Decision Trees

This module implements:
- K-Means Clustering
- Hierarchical Clustering
- Decision Tree Classification

Author: Student
Course: Intelligent Data Analysis (KhNURE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.datasets import make_blobs, load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Set style - professional academic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
})
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6B4C9A']
sns.set_palette(COLORS)

# Results directory
import os
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==================== CLUSTERING ====================

def generate_clustering_data(n_samples=300, n_features=2, n_clusters=4,
                            cluster_std=1.0, random_state=42):
    """Generate sample data for clustering."""
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return X, y_true


def kmeans_clustering(X, n_clusters=3, random_state=42):
    """
    Perform K-Means clustering.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to cluster
    n_clusters : int
        Number of clusters

    Returns:
    --------
    dict : Results containing labels, centroids, and metrics
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    # Calculate metrics
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    inertia = kmeans.inertia_

    return {
        'model': kmeans,
        'labels': labels,
        'centroids': kmeans.cluster_centers_,
        'silhouette_score': silhouette,
        'calinski_harabasz_score': calinski,
        'inertia': inertia,
        'n_clusters': n_clusters
    }


def find_optimal_k(X, k_range=range(2, 11)):
    """
    Find optimal number of clusters using elbow method and silhouette score.

    Parameters:
    -----------
    X : array-like
        Data to cluster
    k_range : range
        Range of k values to test

    Returns:
    --------
    dict : Results containing inertias and silhouette scores
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, kmeans.labels_))

    return {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouettes': silhouettes
    }


def plot_elbow_and_silhouette(optimal_k_results, save_path=None):
    """Plot elbow curve and silhouette scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_range = optimal_k_results['k_range']

    # Elbow curve
    ax1 = axes[0]
    ax1.plot(k_range, optimal_k_results['inertias'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)

    # Silhouette scores
    ax2 = axes[1]
    ax2.plot(k_range, optimal_k_results['silhouettes'], 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score Method')
    ax2.grid(True, alpha=0.3)

    # Mark best silhouette
    best_k_idx = np.argmax(optimal_k_results['silhouettes'])
    best_k = k_range[best_k_idx]
    ax2.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def plot_clusters(X, labels, centroids=None, title='Cluster Analysis',
                  save_path=None):
    """Plot clustering results."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                        alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X',
                  s=200, edgecolors='black', linewidth=2, label='Centroids')
        ax.legend()

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    plt.colorbar(scatter, label='Cluster')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def hierarchical_clustering(X, n_clusters=3, linkage_method='ward'):
    """
    Perform Hierarchical (Agglomerative) clustering.

    Parameters:
    -----------
    X : array-like
        Data to cluster
    n_clusters : int
        Number of clusters
    linkage_method : str
        Linkage method ('ward', 'complete', 'average', 'single')

    Returns:
    --------
    dict : Results containing labels and metrics
    """
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method
    )
    labels = clustering.fit_predict(X)

    # Calculate metrics
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)

    return {
        'model': clustering,
        'labels': labels,
        'silhouette_score': silhouette,
        'calinski_harabasz_score': calinski,
        'n_clusters': n_clusters,
        'linkage_method': linkage_method
    }


def plot_dendrogram(X, method='ward', max_d=None, save_path=None):
    """Plot hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate linkage
    Z = linkage(X, method=method)

    # Plot dendrogram
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=8)

    if max_d:
        ax.axhline(y=max_d, color='r', linestyle='--', label=f'Cut at {max_d}')
        ax.legend()

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Distance')
    ax.set_title(f'Hierarchical Clustering Dendrogram (method={method})')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def dbscan_clustering(X, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering.

    Parameters:
    -----------
    X : array-like
        Data to cluster
    eps : float
        Maximum distance between two samples
    min_samples : int
        Minimum samples in a neighborhood

    Returns:
    --------
    dict : Results containing labels and metrics
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    # Number of clusters (excluding noise labeled as -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Calculate metrics only if more than 1 cluster
    if n_clusters > 1:
        # Exclude noise points for silhouette calculation
        mask = labels != -1
        if mask.sum() > 0 and len(set(labels[mask])) > 1:
            silhouette = silhouette_score(X[mask], labels[mask])
        else:
            silhouette = None
    else:
        silhouette = None

    return {
        'model': dbscan,
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette_score': silhouette,
        'eps': eps,
        'min_samples': min_samples
    }


# ==================== DECISION TREES ====================

def load_sample_classification_data():
    """Load Iris dataset for classification."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    target_names = iris.target_names
    return X, y, target_names


def decision_tree_classification(X, y, max_depth=None, min_samples_split=2,
                                 test_size=0.3, random_state=42):
    """
    Train a Decision Tree classifier.

    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target labels
    max_depth : int or None
        Maximum depth of tree
    min_samples_split : int
        Minimum samples to split
    test_size : float
        Test set proportion
    random_state : int
        Random seed

    Returns:
    --------
    dict : Results containing model, predictions, and metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train model
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    dt.fit(X_train, y_train)

    # Predictions
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)

    # Metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    # Cross-validation
    cv_scores = cross_val_score(dt, X, y, cv=5)

    return {
        'model': dt,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importances': dict(zip(
            X.columns if hasattr(X, 'columns') else range(X.shape[1]),
            dt.feature_importances_
        ))
    }


def plot_decision_tree(dt_results, feature_names=None, class_names=None,
                       save_path=None):
    """Visualize decision tree."""
    fig, ax = plt.subplots(figsize=(20, 12))

    plot_tree(
        dt_results['model'],
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )

    ax.set_title('Decision Tree Visualization')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def plot_feature_importance(dt_results, save_path=None):
    """Plot feature importances."""
    importances = dt_results['feature_importances']

    fig, ax = plt.subplots(figsize=(10, 6))

    features = list(importances.keys())
    values = list(importances.values())

    # Sort by importance
    sorted_idx = np.argsort(values)
    features = [features[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]

    ax.barh(range(len(features)), values, align='center')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importances')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def bmw_customer_example():
    """
    Example from the lab: BMW dealership customer classification.

    Predict likelihood of customer buying BMW M5 based on:
    - Age
    - Income level
    - Number of cars owned
    - Family status
    - Children
    - Home ownership
    """
    np.random.seed(42)
    n_samples = 200

    # Generate synthetic BMW customer data
    data = pd.DataFrame({
        'age': np.random.randint(22, 65, n_samples),
        'income': np.random.choice(['low', 'medium', 'high'], n_samples,
                                   p=[0.3, 0.4, 0.3]),
        'cars_owned': np.random.randint(0, 5, n_samples),
        'married': np.random.choice([0, 1], n_samples),
        'has_children': np.random.choice([0, 1], n_samples),
        'owns_home': np.random.choice([0, 1], n_samples)
    })

    # Create target: will buy M5 (simplified logic)
    data['will_buy_m5'] = (
        (data['income'] == 'high') &
        (data['age'] >= 35) &
        (data['age'] <= 55) &
        (data['cars_owned'] >= 1)
    ).astype(int)

    # Add some noise
    noise_idx = np.random.choice(n_samples, size=20, replace=False)
    data.loc[noise_idx, 'will_buy_m5'] = 1 - data.loc[noise_idx, 'will_buy_m5']

    # Encode categorical variables
    le = LabelEncoder()
    data['income_encoded'] = le.fit_transform(data['income'])

    # Features and target
    feature_cols = ['age', 'income_encoded', 'cars_owned', 'married',
                    'has_children', 'owns_home']
    X = data[feature_cols]
    y = data['will_buy_m5']

    return X, y, data, feature_cols


def run_full_analysis():
    """Run complete clustering and classification analysis."""
    print("=" * 60)
    print("LAB 2: CLUSTERING AND DECISION TREES")
    print("=" * 60)

    # Part 1: K-Means Clustering
    print("\n" + "=" * 60)
    print("PART 1: K-Means Clustering")
    print("=" * 60)

    X_cluster, y_true = generate_clustering_data(n_clusters=4)
    print(f"\nGenerated {len(X_cluster)} samples with 4 clusters")

    # Find optimal k
    optimal_k = find_optimal_k(X_cluster)
    plot_elbow_and_silhouette(optimal_k, save_path=os.path.join(RESULTS_DIR, 'elbow_silhouette.png'))

    # Perform K-Means with k=4
    kmeans_results = kmeans_clustering(X_cluster, n_clusters=4)
    print(f"\nK-Means Results (k=4):")
    print(f"  Silhouette Score: {kmeans_results['silhouette_score']:.4f}")
    print(f"  Calinski-Harabasz Score: {kmeans_results['calinski_harabasz_score']:.2f}")
    print(f"  Inertia: {kmeans_results['inertia']:.2f}")

    plot_clusters(
        X_cluster, kmeans_results['labels'], kmeans_results['centroids'],
        title='K-Means Clustering (k=4)',
        save_path=os.path.join(RESULTS_DIR, 'kmeans_clusters.png')
    )

    # Part 2: Hierarchical Clustering
    print("\n" + "=" * 60)
    print("PART 2: Hierarchical Clustering")
    print("=" * 60)

    hier_results = hierarchical_clustering(X_cluster, n_clusters=4)
    print(f"\nHierarchical Clustering Results:")
    print(f"  Silhouette Score: {hier_results['silhouette_score']:.4f}")
    print(f"  Linkage Method: {hier_results['linkage_method']}")

    # Plot dendrogram (with subset for clarity)
    X_sample = X_cluster[:50]
    plot_dendrogram(X_sample, method='ward', save_path=os.path.join(RESULTS_DIR, 'dendrogram.png'))

    plot_clusters(
        X_cluster, hier_results['labels'],
        title='Hierarchical Clustering (Ward linkage)',
        save_path=os.path.join(RESULTS_DIR, 'hierarchical_clusters.png')
    )

    # Part 3: DBSCAN
    print("\n" + "=" * 60)
    print("PART 3: DBSCAN Clustering")
    print("=" * 60)

    dbscan_results = dbscan_clustering(X_cluster, eps=1.0, min_samples=5)
    print(f"\nDBSCAN Results:")
    print(f"  Number of Clusters: {dbscan_results['n_clusters']}")
    print(f"  Number of Noise Points: {dbscan_results['n_noise']}")
    if dbscan_results['silhouette_score']:
        print(f"  Silhouette Score: {dbscan_results['silhouette_score']:.4f}")

    plot_clusters(
        X_cluster, dbscan_results['labels'],
        title=f'DBSCAN Clustering (eps={dbscan_results["eps"]})',
        save_path=os.path.join(RESULTS_DIR, 'dbscan_clusters.png')
    )

    # Part 4: Decision Trees
    print("\n" + "=" * 60)
    print("PART 4: Decision Tree Classification")
    print("=" * 60)

    X_iris, y_iris, target_names = load_sample_classification_data()
    print(f"\nIris Dataset: {len(X_iris)} samples, {X_iris.shape[1]} features")

    dt_results = decision_tree_classification(X_iris, y_iris, max_depth=4)

    print(f"\nDecision Tree Results:")
    print(f"  Training Accuracy: {dt_results['train_accuracy']:.4f}")
    print(f"  Test Accuracy: {dt_results['test_accuracy']:.4f}")
    print(f"  Cross-Validation: {dt_results['cv_mean']:.4f} (+/- {dt_results['cv_std']:.4f})")

    print("\nFeature Importances:")
    for feature, importance in sorted(dt_results['feature_importances'].items(),
                                       key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")

    # Visualizations
    plot_decision_tree(
        dt_results,
        feature_names=list(X_iris.columns),
        class_names=list(target_names),
        save_path=os.path.join(RESULTS_DIR, 'decision_tree.png')
    )

    plot_confusion_matrix(
        dt_results['y_test'], dt_results['y_pred_test'],
        class_names=list(target_names),
        save_path=os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    )

    plot_feature_importance(dt_results, save_path=os.path.join(RESULTS_DIR, 'feature_importance.png'))

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(
        dt_results['y_test'], dt_results['y_pred_test'],
        target_names=target_names
    ))

    # Tree rules in text format
    print("\nDecision Tree Rules:")
    tree_rules = export_text(dt_results['model'],
                            feature_names=list(X_iris.columns))
    print(tree_rules)

    # Part 5: BMW Customer Example
    print("\n" + "=" * 60)
    print("PART 5: BMW Customer Classification Example")
    print("=" * 60)

    X_bmw, y_bmw, bmw_data, bmw_features = bmw_customer_example()
    print(f"\nBMW Customer Data: {len(X_bmw)} customers")
    print(f"Features: {bmw_features}")
    print(f"Will buy M5: {y_bmw.sum()} customers ({100*y_bmw.mean():.1f}%)")

    bmw_dt_results = decision_tree_classification(X_bmw, y_bmw, max_depth=3)
    print(f"\nBMW Decision Tree Results:")
    print(f"  Test Accuracy: {bmw_dt_results['test_accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_analysis()
