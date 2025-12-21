# Lab Work #3: Association Rules - Apriori Algorithm

## Objective
Learn association rule mining using the Apriori algorithm:
- Frequent itemset mining
- Association rule generation
- Market basket analysis

## Theoretical Background

### Association Rules
An association rule is an implication of the form **X → Y**, where:
- **X** (antecedent): Items that appear together
- **Y** (consequent): Items implied by X

### Key Metrics

#### Support
Proportion of transactions containing the itemset:
```
support(X → Y) = support(X ∪ Y) = |transactions containing X and Y| / |total transactions|
```

#### Confidence
Probability of Y given X:
```
confidence(X → Y) = support(X ∪ Y) / support(X)
```

#### Lift
How much more likely Y is given X compared to random:
```
lift(X → Y) = confidence(X → Y) / support(Y)
```
- Lift > 1: Positive correlation
- Lift = 1: Independent
- Lift < 1: Negative correlation

### Anti-monotonicity Property
If itemset X is not frequent, no superset of X can be frequent.
This property enables efficient pruning in the Apriori algorithm.

### Apriori Algorithm Steps
1. Generate 1-itemsets with support ≥ min_support
2. Generate k-itemsets from (k-1)-itemsets
3. Prune itemsets with support < min_support
4. Repeat until no more itemsets can be generated
5. Generate rules from frequent itemsets

## Implementation

### Files
- `apriori_analysis.py` - Main implementation using mlxtend

### Usage
```bash
source venv/bin/activate
cd labs/lab3_apriori
python apriori_analysis.py
```

### Example: Market Basket Analysis
Classic example of Diapers → Beer association:
```
Rule: Diapers → Beer
  Support: 0.15
  Confidence: 0.60
  Lift: 1.8
```

## Tasks Completed

1. ✅ Frequent itemset mining with Apriori
2. ✅ Association rule generation
3. ✅ Support, confidence, and lift calculation
4. ✅ Sensitivity analysis (varying thresholds)
5. ✅ Co-occurrence matrix
6. ✅ Network graph visualization
7. ✅ Anti-monotonicity demonstration

## Output Files
- `support_distribution.png` - Support value histogram
- `rules_metrics.png` - Support vs Confidence scatter plot
- `rules_heatmap.png` - Top rules heatmap
- `rules_network.png` - Association rules network graph
- `frequency_matrix.png` - Item co-occurrence matrix
- `sensitivity_analysis.png` - Threshold sensitivity heatmap

## Dependencies
- numpy, pandas
- mlxtend
- networkx
- matplotlib, seaborn
