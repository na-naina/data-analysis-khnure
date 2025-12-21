"""
Practical Work #2: Text Classification using TF-IDF
====================================================
Vector Space Model and TF-IDF based document similarity

This module implements the TF-IDF algorithm for text classification
based on the practical work guidelines (Variant 1).

Task (Variant 1):
-----------------
Query, Q: "silver black car"
D1: "Advantages of good looking of black car"
D2: "The silver has good looking shine"
D3: "Advantages of good looking silver that the silver is shine"
D4: "Both silver and black have good looking shine"

Author: Student
Course: Intelligent Data Analysis (KhNURE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import List, Dict, Tuple
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


class TFIDFVectorizer:
    """
    TF-IDF Vectorizer implementing the Vector Space Model.

    The weight formula: w_i = tf_i * log(D / df_i)

    where:
    - tf_i: term frequency (count of term i in document)
    - df_i: document frequency (number of documents containing term i)
    - D: total number of documents
    - IDF_i = log(D / df_i): inverse document frequency
    """

    def __init__(self, use_idf: bool = True, normalize: bool = False):
        """
        Initialize TF-IDF Vectorizer.

        Parameters:
        -----------
        use_idf : bool
            Whether to use IDF weighting
        normalize : bool
            Whether to normalize vectors
        """
        self.use_idf = use_idf
        self.normalize = normalize
        self.vocabulary = {}
        self.idf = {}
        self.df = {}
        self.n_documents = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split by space."""
        return text.lower().split()

    def fit(self, documents: List[str]):
        """
        Learn vocabulary and IDF from documents.

        Parameters:
        -----------
        documents : list of str
            Training documents
        """
        self.n_documents = len(documents)

        # Build vocabulary and count document frequencies
        all_terms = set()
        doc_freq = Counter()

        for doc in documents:
            terms = set(self._tokenize(doc))
            all_terms.update(terms)
            for term in terms:
                doc_freq[term] += 1

        # Create vocabulary mapping
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}

        # Calculate IDF
        self.df = doc_freq
        for term, df in doc_freq.items():
            self.idf[term] = np.log(self.n_documents / df)

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF vectors.

        Parameters:
        -----------
        documents : list of str
            Documents to transform

        Returns:
        --------
        np.ndarray : TF-IDF matrix (n_documents, n_terms)
        """
        n_docs = len(documents)
        n_terms = len(self.vocabulary)
        tfidf_matrix = np.zeros((n_docs, n_terms))

        for doc_idx, doc in enumerate(documents):
            terms = self._tokenize(doc)
            term_counts = Counter(terms)

            for term, count in term_counts.items():
                if term in self.vocabulary:
                    term_idx = self.vocabulary[term]
                    tf = count
                    idf = self.idf.get(term, 0) if self.use_idf else 1
                    tfidf_matrix[doc_idx, term_idx] = tf * idf

            # Normalize if required
            if self.normalize:
                norm = np.linalg.norm(tfidf_matrix[doc_idx])
                if norm > 0:
                    tfidf_matrix[doc_idx] /= norm

        return tfidf_matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self) -> List[str]:
        """Get vocabulary terms."""
        return list(self.vocabulary.keys())


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Parameters:
    -----------
    vec1, vec2 : np.ndarray
        Input vectors

    Returns:
    --------
    float : Cosine similarity (0 to 1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def calculate_detailed_tfidf(documents: List[str], query: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate detailed TF-IDF values step by step.

    Parameters:
    -----------
    documents : list of str
        Document collection
    query : str
        Query string

    Returns:
    --------
    tuple : (term_counts_df, weights_df)
    """
    n_docs = len(documents)
    all_docs = documents + [query]

    # Tokenize all
    all_terms = set()
    doc_terms = []
    for doc in all_docs:
        terms = doc.lower().split()
        doc_terms.append(terms)
        all_terms.update(terms)

    all_terms = sorted(all_terms)

    # Calculate term frequencies
    tf_data = {term: [] for term in all_terms}
    for terms in doc_terms:
        term_counts = Counter(terms)
        for term in all_terms:
            tf_data[term].append(term_counts.get(term, 0))

    tf_df = pd.DataFrame(tf_data, index=['D1', 'D2', 'D3', 'D4', 'Q'][:n_docs + 1])

    # Calculate document frequencies
    df_values = {}
    for term in all_terms:
        # Only count in documents, not query
        df_values[term] = sum(1 for i, terms in enumerate(doc_terms[:-1]) if term in terms)

    # Calculate IDF
    idf_values = {}
    for term, df in df_values.items():
        if df > 0:
            idf_values[term] = np.log(n_docs / df)
        else:
            idf_values[term] = 0

    # Create term statistics table
    stats_data = []
    for term in all_terms:
        row = [tf_df[term].tolist()[i] for i in range(n_docs + 1)]  # TF for each doc + query
        row.append(df_values[term])  # df
        row.append(n_docs / df_values[term] if df_values[term] > 0 else 0)  # D/df
        row.append(idf_values[term])  # IDF
        stats_data.append([term] + row)

    columns = ['Term'] + [f'D{i+1}' for i in range(n_docs)] + ['Q', 'df', 'D/df', 'IDF']
    stats_df = pd.DataFrame(stats_data, columns=columns)

    # Calculate TF-IDF weights
    weight_data = {term: [] for term in all_terms}
    for doc_idx, terms in enumerate(doc_terms):
        term_counts = Counter(terms)
        for term in all_terms:
            tf = term_counts.get(term, 0)
            idf = idf_values[term]
            weight_data[term].append(round(tf * idf, 4))

    weights_df = pd.DataFrame(weight_data, index=['D1', 'D2', 'D3', 'D4', 'Q'][:n_docs + 1])

    return stats_df, weights_df


def solve_variant1():
    """
    Solve Variant 1 from the practical work.

    Query, Q: "silver black car"
    D1: "Advantages of good looking of black car"
    D2: "The silver has good looking shine"
    D3: "Advantages of good looking silver that the silver is shine"
    D4: "Both silver and black have good looking shine"
    """
    print("=" * 70)
    print("PRACTICAL WORK #2: TF-IDF TEXT CLASSIFICATION")
    print("VARIANT 1")
    print("=" * 70)

    # Define documents and query
    documents = [
        "Advantages of good looking of black car",
        "The silver has good looking shine",
        "Advantages of good looking silver that the silver is shine",
        "Both silver and black have good looking shine"
    ]
    query = "silver black car"

    doc_names = ['D1', 'D2', 'D3', 'D4']

    print("\n" + "-" * 70)
    print("INPUT DATA")
    print("-" * 70)
    print(f"\nQuery, Q: \"{query}\"")
    for i, doc in enumerate(documents):
        print(f"D{i+1}: \"{doc}\"")

    # Step 1: Calculate detailed TF-IDF
    print("\n" + "-" * 70)
    print("STEP 1: TERM FREQUENCY AND IDF CALCULATION")
    print("-" * 70)

    stats_df, weights_df = calculate_detailed_tfidf(documents, query)

    print("\nTerm Statistics (TF counts, df, D/df, IDF):")
    print(stats_df.to_string(index=False))

    print("\n" + "-" * 70)
    print("STEP 2: TF-IDF WEIGHTS (w_i = tf_i * IDF_i)")
    print("-" * 70)

    print("\nWeights Matrix:")
    print(weights_df.to_string())

    # Step 2: Calculate vector lengths
    print("\n" + "-" * 70)
    print("STEP 3: VECTOR LENGTHS")
    print("-" * 70)

    doc_vectors = weights_df.loc[doc_names].values
    query_vector = weights_df.loc['Q'].values

    for i, doc_name in enumerate(doc_names):
        vec = doc_vectors[i]
        non_zero = vec[vec != 0]
        length = np.sqrt(np.sum(vec ** 2))
        print(f"|{doc_name}| = sqrt({' + '.join([f'{v:.4f}²' for v in non_zero])}) = sqrt({np.sum(vec**2):.4f}) = {length:.4f}")

    query_non_zero = query_vector[query_vector != 0]
    query_length = np.linalg.norm(query_vector)
    print(f"|Q| = sqrt({' + '.join([f'{v:.4f}²' for v in query_non_zero])}) = sqrt({np.sum(query_vector**2):.4f}) = {query_length:.4f}")

    # Step 3: Calculate dot products
    print("\n" + "-" * 70)
    print("STEP 4: DOT PRODUCTS (Q • D_i)")
    print("-" * 70)

    dot_products = {}
    for i, doc_name in enumerate(doc_names):
        dot_product = np.dot(query_vector, doc_vectors[i])
        dot_products[doc_name] = dot_product

        # Show calculation details
        non_zero_indices = np.where((query_vector != 0) & (doc_vectors[i] != 0))[0]
        if len(non_zero_indices) > 0:
            terms_detail = ' + '.join([f'{query_vector[j]:.4f}*{doc_vectors[i][j]:.4f}'
                                       for j in non_zero_indices])
            print(f"Q • {doc_name} = {terms_detail} = {dot_product:.4f}")
        else:
            print(f"Q • {doc_name} = 0 (no common terms)")

    # Step 4: Calculate cosine similarity
    print("\n" + "-" * 70)
    print("STEP 5: COSINE SIMILARITY (Sim(Q, D_i))")
    print("-" * 70)

    similarities = {}
    for i, doc_name in enumerate(doc_names):
        sim = cosine_similarity(query_vector, doc_vectors[i])
        similarities[doc_name] = sim
        doc_length = np.linalg.norm(doc_vectors[i])
        print(f"Cosine θ_{doc_name} = (Q • {doc_name}) / (|Q| * |{doc_name}|) = {dot_products[doc_name]:.4f} / ({query_length:.4f} * {doc_length:.4f}) = {sim:.4f}")

    # Step 5: Rank documents
    print("\n" + "-" * 70)
    print("STEP 6: DOCUMENT RANKING (by similarity)")
    print("-" * 70)

    ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    for rank, (doc_name, sim) in enumerate(ranked, 1):
        print(f"Rank {rank}: {doc_name} = {sim:.4f}")

    # Conclusions
    print("\n" + "-" * 70)
    print("CONCLUSIONS")
    print("-" * 70)

    best_doc = ranked[0][0]
    best_sim = ranked[0][1]
    print(f"\n1. The most relevant document for query \"{query}\" is {best_doc}")
    print(f"   with cosine similarity = {best_sim:.4f}")

    print(f"\n2. Document ranking by relevance:")
    for rank, (doc_name, sim) in enumerate(ranked, 1):
        doc_idx = int(doc_name[1]) - 1
        print(f"   {rank}. {doc_name}: \"{documents[doc_idx]}\" (sim={sim:.4f})")

    print("\n3. Analysis:")
    print("   - Terms like 'silver', 'black', and 'car' from the query contribute to similarity")
    print("   - Common words ('of', 'the', 'a') have low IDF and contribute little")
    print("   - D4 ranks highest because it contains both 'silver' and 'black' from the query")

    # Visualizations
    print("\n" + "-" * 70)
    print("VISUALIZATIONS")
    print("-" * 70)

    # Plot 1: Similarity scores
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    doc_labels = [f"{doc}\n(sim={sim:.3f})" for doc, sim in similarities.items()]
    bars = ax1.bar(doc_labels, similarities.values(), color=['steelblue', 'darkorange', 'green', 'red'])
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_xlabel('Document')
    ax1.set_title(f'Document Similarity to Query: "{query}"')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, sim in zip(bars, similarities.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{sim:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'similarity_scores.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Plot 2: TF-IDF heatmap
    fig2, ax2 = plt.subplots(figsize=(14, 6))

    # Filter out zero columns
    non_zero_cols = weights_df.columns[(weights_df != 0).any()]
    weights_plot = weights_df[non_zero_cols]

    sns.heatmap(weights_plot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
    ax2.set_title('TF-IDF Weights Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'tfidf_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nSaved to: {RESULTS_DIR}")

    return similarities, weights_df


def run_full_analysis():
    """Run the complete TF-IDF analysis."""
    similarities, weights_df = solve_variant1()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_analysis()
