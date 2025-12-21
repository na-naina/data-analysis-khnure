"""
Lab Work #3: Association Rules - Apriori Algorithm
===================================================
Market Basket Analysis using Apriori Algorithm

This module implements:
- Apriori algorithm for frequent itemset mining
- Association rule generation
- Support, confidence, and lift metrics

Author: Student
Course: Intelligent Data Analysis (KhNURE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
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


def create_sample_transactions():
    """
    Create sample transaction data for market basket analysis.
    Based on the lab manual example.
    """
    transactions = [
        ['Bread', 'Milk'],
        ['Bread', 'Diapers', 'Beer', 'Eggs'],
        ['Milk', 'Diapers', 'Beer', 'Cola'],
        ['Bread', 'Milk', 'Diapers', 'Beer'],
        ['Bread', 'Milk', 'Diapers', 'Cola'],
        ['Bread', 'Milk', 'Beer'],
        ['Bread', 'Diapers', 'Beer'],
        ['Bread', 'Milk', 'Diapers', 'Beer', 'Eggs'],
        ['Bread', 'Milk', 'Cola'],
        ['Bread', 'Diapers', 'Cola']
    ]
    return transactions


def create_supermarket_transactions():
    """
    Create larger supermarket transaction dataset.
    """
    np.random.seed(42)

    # Define item categories and items
    items = {
        'dairy': ['Milk', 'Cheese', 'Yogurt', 'Butter'],
        'bakery': ['Bread', 'Cookies', 'Cake'],
        'beverages': ['Cola', 'Beer', 'Juice', 'Water'],
        'snacks': ['Chips', 'Candy'],
        'household': ['Diapers', 'Detergent', 'Paper Towels'],
        'proteins': ['Eggs', 'Chicken', 'Fish']
    }

    all_items = [item for category in items.values() for item in category]

    # Generate transactions
    n_transactions = 100
    transactions = []

    for _ in range(n_transactions):
        # Random number of items per transaction
        n_items = np.random.randint(2, 8)
        # Select items with some patterns
        transaction = []

        # People who buy diapers often buy beer (classic association)
        if np.random.random() < 0.3:
            transaction.extend(['Diapers', 'Beer'])

        # Bread and milk are common together
        if np.random.random() < 0.5:
            transaction.extend(['Bread', 'Milk'])

        # Fill remaining slots with random items
        remaining = n_items - len(transaction)
        if remaining > 0:
            available = [i for i in all_items if i not in transaction]
            transaction.extend(np.random.choice(available, min(remaining, len(available)),
                                                 replace=False).tolist())

        transactions.append(list(set(transaction)))  # Remove duplicates

    return transactions


def transactions_to_dataframe(transactions):
    """
    Convert list of transactions to one-hot encoded DataFrame.

    Parameters:
    -----------
    transactions : list of lists
        Each inner list contains items in a transaction

    Returns:
    --------
    DataFrame : One-hot encoded transaction matrix
    """
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)
    return df


def run_apriori(df, min_support=0.1, use_colnames=True):
    """
    Run Apriori algorithm to find frequent itemsets.

    Parameters:
    -----------
    df : DataFrame
        One-hot encoded transaction data
    min_support : float
        Minimum support threshold (0 to 1)
    use_colnames : bool
        Use column names instead of indices

    Returns:
    --------
    DataFrame : Frequent itemsets with support values
    """
    frequent_itemsets = apriori(
        df,
        min_support=min_support,
        use_colnames=use_colnames
    )
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
    return frequent_itemsets


def generate_rules(frequent_itemsets, min_confidence=0.5, min_lift=1.0):
    """
    Generate association rules from frequent itemsets.

    Parameters:
    -----------
    frequent_itemsets : DataFrame
        Output from apriori algorithm
    min_confidence : float
        Minimum confidence threshold (0 to 1)
    min_lift : float
        Minimum lift threshold

    Returns:
    --------
    DataFrame : Association rules with metrics
    """
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    # Filter by lift
    rules = rules[rules['lift'] >= min_lift]

    # Sort by confidence
    rules = rules.sort_values('confidence', ascending=False)

    return rules


def print_rules(rules, top_n=10):
    """Print association rules in readable format."""
    print(f"\nTop {min(top_n, len(rules))} Association Rules:")
    print("-" * 80)

    for idx, row in rules.head(top_n).iterrows():
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        print(f"Rule: {antecedents} → {consequents}")
        print(f"  Support: {row['support']:.4f}")
        print(f"  Confidence: {row['confidence']:.4f}")
        print(f"  Lift: {row['lift']:.4f}")
        print()


def plot_support_distribution(frequent_itemsets, save_path=None):
    """Plot distribution of support values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of support values
    ax1 = axes[0]
    ax1.hist(frequent_itemsets['support'], bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Support')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Support Values')
    ax1.grid(True, alpha=0.3)

    # Support by itemset length
    ax2 = axes[1]
    support_by_length = frequent_itemsets.groupby('length')['support'].mean()
    ax2.bar(support_by_length.index, support_by_length.values, edgecolor='black')
    ax2.set_xlabel('Itemset Length')
    ax2.set_ylabel('Average Support')
    ax2.set_title('Average Support by Itemset Length')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def plot_rules_metrics(rules, save_path=None):
    """Plot scatter plot of confidence vs support with lift as color."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        rules['support'],
        rules['confidence'],
        c=rules['lift'],
        cmap='viridis',
        s=100,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )

    ax.set_xlabel('Support')
    ax.set_ylabel('Confidence')
    ax.set_title('Association Rules: Support vs Confidence (color = Lift)')
    plt.colorbar(scatter, label='Lift')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def plot_rules_heatmap(rules, top_n=20, save_path=None):
    """Plot heatmap of top rules."""
    # Get top rules
    top_rules = rules.head(top_n)

    # Create rule labels
    rule_labels = []
    for _, row in top_rules.iterrows():
        ant = ', '.join(sorted(row['antecedents']))
        con = ', '.join(sorted(row['consequents']))
        rule_labels.append(f"{ant} → {con}")

    # Create matrix for heatmap
    metrics_df = pd.DataFrame({
        'Support': top_rules['support'].values,
        'Confidence': top_rules['confidence'].values,
        'Lift': top_rules['lift'].values
    }, index=rule_labels)

    # Normalize for visualization
    metrics_normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())

    fig, ax = plt.subplots(figsize=(10, max(8, len(rule_labels) * 0.4)))

    sns.heatmap(
        metrics_normalized,
        annot=metrics_df.round(3),
        fmt='',
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Normalized Value'}
    )

    ax.set_title(f'Top {len(rule_labels)} Association Rules')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def plot_rules_network(rules, top_n=15, save_path=None):
    """Plot network graph of association rules."""
    fig, ax = plt.subplots(figsize=(14, 10))

    G = nx.DiGraph()

    # Add edges for top rules
    for _, row in rules.head(top_n).iterrows():
        for ant in row['antecedents']:
            for con in row['consequents']:
                G.add_edge(ant, con, weight=row['confidence'], lift=row['lift'])

    # Calculate node sizes based on frequency
    node_sizes = {}
    for node in G.nodes():
        node_sizes[node] = sum(1 for _, row in rules.head(top_n).iterrows()
                              if node in row['antecedents'] or node in row['consequents'])

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[node_sizes[node] * 500 for node in G.nodes()],
        node_color='lightblue',
        edgecolors='navy',
        linewidths=2,
        ax=ax
    )

    # Draw edges with width based on confidence
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos,
        width=edge_weights,
        alpha=0.6,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
        ax=ax
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    ax.set_title('Association Rules Network Graph')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def create_frequency_matrix(transactions):
    """
    Create item co-occurrence frequency matrix.

    Parameters:
    -----------
    transactions : list of lists
        Transaction data

    Returns:
    --------
    DataFrame : Co-occurrence matrix
    """
    df = transactions_to_dataframe(transactions)
    items = df.columns.tolist()

    # Calculate co-occurrence
    cooccurrence = pd.DataFrame(
        np.zeros((len(items), len(items))),
        index=items,
        columns=items
    )

    for _, row in df.iterrows():
        present_items = [item for item in items if row[item]]
        for i, item1 in enumerate(present_items):
            for item2 in present_items[i:]:
                cooccurrence.loc[item1, item2] += 1
                if item1 != item2:
                    cooccurrence.loc[item2, item1] += 1

    return cooccurrence


def plot_frequency_matrix(cooccurrence, save_path=None):
    """Plot item frequency/co-occurrence matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Mask diagonal for better visualization
    mask = np.eye(len(cooccurrence), dtype=bool)

    sns.heatmap(
        cooccurrence,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        mask=mask,
        ax=ax,
        square=True
    )

    ax.set_title('Item Co-occurrence Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def sensitivity_analysis(df, support_range=None, confidence_range=None):
    """
    Analyze how different thresholds affect number of rules.

    Parameters:
    -----------
    df : DataFrame
        One-hot encoded transaction data
    support_range : list
        Support thresholds to test
    confidence_range : list
        Confidence thresholds to test

    Returns:
    --------
    dict : Results of sensitivity analysis
    """
    if support_range is None:
        support_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    if confidence_range is None:
        confidence_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = []

    for min_sup in support_range:
        for min_conf in confidence_range:
            try:
                freq_items = apriori(df, min_support=min_sup, use_colnames=True)
                if len(freq_items) > 0:
                    rules = association_rules(
                        freq_items, metric="confidence", min_threshold=min_conf
                    )
                    n_rules = len(rules)
                else:
                    n_rules = 0
            except:
                n_rules = 0

            results.append({
                'min_support': min_sup,
                'min_confidence': min_conf,
                'n_rules': n_rules
            })

    return pd.DataFrame(results)


def plot_sensitivity_analysis(sensitivity_df, save_path=None):
    """Plot sensitivity analysis results."""
    fig, ax = plt.subplots(figsize=(10, 8))

    pivot_table = sensitivity_df.pivot(
        index='min_support',
        columns='min_confidence',
        values='n_rules'
    )

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.0f',
        cmap='YlGnBu',
        ax=ax,
        cbar_kws={'label': 'Number of Rules'}
    )

    ax.set_xlabel('Minimum Confidence')
    ax.set_ylabel('Minimum Support')
    ax.set_title('Sensitivity Analysis: Number of Rules by Thresholds')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def run_full_analysis():
    """Run complete Apriori analysis demonstration."""
    print("=" * 60)
    print("LAB 3: ASSOCIATION RULES - APRIORI ALGORITHM")
    print("=" * 60)

    # Part 1: Simple Example
    print("\n" + "=" * 60)
    print("PART 1: Simple Transaction Example")
    print("=" * 60)

    transactions = create_sample_transactions()
    print(f"\nNumber of transactions: {len(transactions)}")
    print("\nSample transactions:")
    for i, t in enumerate(transactions[:5], 1):
        print(f"  Transaction {i}: {t}")

    # Convert to DataFrame
    df = transactions_to_dataframe(transactions)
    print(f"\nTransaction matrix shape: {df.shape}")

    # Run Apriori
    frequent_itemsets = run_apriori(df, min_support=0.2)
    print(f"\nFrequent itemsets found: {len(frequent_itemsets)}")
    print("\nTop frequent itemsets:")
    print(frequent_itemsets.sort_values('support', ascending=False).head(10).to_string())

    # Generate rules
    rules = generate_rules(frequent_itemsets, min_confidence=0.5, min_lift=1.0)
    print(f"\nAssociation rules found: {len(rules)}")
    print_rules(rules, top_n=10)

    # Part 2: Larger Dataset
    print("\n" + "=" * 60)
    print("PART 2: Supermarket Transaction Analysis")
    print("=" * 60)

    supermarket_transactions = create_supermarket_transactions()
    print(f"\nNumber of transactions: {len(supermarket_transactions)}")

    df_super = transactions_to_dataframe(supermarket_transactions)
    print(f"Number of unique items: {df_super.shape[1]}")

    # Item frequencies
    item_freq = df_super.sum().sort_values(ascending=False)
    print("\nTop 10 most frequent items:")
    for item, count in item_freq.head(10).items():
        print(f"  {item}: {count} ({100*count/len(df_super):.1f}%)")

    # Run Apriori
    freq_items_super = run_apriori(df_super, min_support=0.1)
    print(f"\nFrequent itemsets found: {len(freq_items_super)}")

    # Generate rules
    rules_super = generate_rules(freq_items_super, min_confidence=0.4, min_lift=1.0)
    print(f"Association rules found: {len(rules_super)}")
    print_rules(rules_super, top_n=10)

    # Part 3: Visualizations
    print("\n" + "=" * 60)
    print("PART 3: Visualizations")
    print("=" * 60)

    plot_support_distribution(freq_items_super, save_path=os.path.join(RESULTS_DIR, 'support_distribution.png'))

    if len(rules_super) > 0:
        plot_rules_metrics(rules_super, save_path=os.path.join(RESULTS_DIR, 'rules_metrics.png'))
        plot_rules_heatmap(rules_super, top_n=15, save_path=os.path.join(RESULTS_DIR, 'rules_heatmap.png'))
        plot_rules_network(rules_super, top_n=15, save_path=os.path.join(RESULTS_DIR, 'rules_network.png'))

    # Co-occurrence matrix
    cooccurrence = create_frequency_matrix(supermarket_transactions)
    plot_frequency_matrix(cooccurrence, save_path=os.path.join(RESULTS_DIR, 'frequency_matrix.png'))

    # Part 4: Sensitivity Analysis
    print("\n" + "=" * 60)
    print("PART 4: Sensitivity Analysis")
    print("=" * 60)

    sensitivity_df = sensitivity_analysis(df_super)
    print("\nNumber of rules by threshold combinations:")
    print(sensitivity_df.pivot(
        index='min_support',
        columns='min_confidence',
        values='n_rules'
    ).to_string())

    plot_sensitivity_analysis(sensitivity_df, save_path=os.path.join(RESULTS_DIR, 'sensitivity_analysis.png'))

    # Part 5: Anti-monotonicity Property
    print("\n" + "=" * 60)
    print("PART 5: Anti-monotonicity Property Demonstration")
    print("=" * 60)

    print("\nAnti-monotonicity: If itemset X is not frequent,")
    print("then no superset of X can be frequent.")
    print("\nItemsets by length:")
    for length in sorted(freq_items_super['length'].unique()):
        count = len(freq_items_super[freq_items_super['length'] == length])
        avg_support = freq_items_super[freq_items_super['length'] == length]['support'].mean()
        print(f"  Length {length}: {count} itemsets, avg support = {avg_support:.4f}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_analysis()
