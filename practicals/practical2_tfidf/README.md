# Practical Work #2: TF-IDF Text Classification

## Objective
Learn text classification using the TF-IDF (Term Frequency - Inverse Document Frequency) algorithm and Vector Space Model.

## Variant 1

### Input Data
```
Query, Q: "silver black car"
D1: "Advantages of good looking of black car"
D2: "The silver has good looking shine"
D3: "Advantages of good looking silver that the silver is shine"
D4: "Both silver and black have good looking shine"
```

### Tasks
1. Calculate vector lengths for each document
2. Calculate query vector
3. Calculate dot products (Q • D_i)
4. Calculate cosine similarity (similarity measure)
5. Sort documents by similarity
6. Draw conclusions

## Theoretical Background

### TF-IDF Formula
```
w_i = tf_i * log(D / df_i)
```

Where:
- **tf_i**: Term frequency (number of times term i appears in document)
- **df_i**: Document frequency (number of documents containing term i)
- **D**: Total number of documents
- **IDF_i = log(D/df_i)**: Inverse Document Frequency

### Cosine Similarity
```
Sim(Q, D_i) = (Q • D_i) / (|Q| * |D_i|)
```

Where:
- **Q • D_i**: Dot product of query and document vectors
- **|Q|, |D_i|**: Vector magnitudes (Euclidean norms)

## Implementation

### Files
- `tfidf_classification.py` - Complete TF-IDF implementation

### Usage
```bash
source venv/bin/activate
cd practicals/practical2_tfidf
python tfidf_classification.py
```

## Results (Variant 1)

### Document Ranking
| Rank | Document | Similarity |
|------|----------|------------|
| 1 | D4 | ~0.68 |
| 2 | D1 | ~0.48 |
| 3 | D3 | ~0.32 |
| 4 | D2 | ~0.25 |

### Conclusion
- D4 ("Both silver and black have good looking shine") is most relevant
- Contains both query terms "silver" and "black"
- Common words contribute less due to low IDF values

## Output Files
- `similarity_scores.png` - Bar chart of document similarities
- `tfidf_heatmap.png` - TF-IDF weight matrix visualization

## Dependencies
- numpy, pandas
- matplotlib, seaborn
