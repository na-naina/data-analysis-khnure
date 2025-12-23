# Intelligent Data Analysis

**Course**: Intelligent Data Analysis (Інтелектуальний аналіз даних)

**University**: Kharkiv National University of Radio Electronics (KhNURE)

**Specialty**: 121 - Software Engineering

**Semester**: 3 (Master's Program)

---

## Project Structure

```
Data Analysis/
├── docs/                          # Assignment specifications (PDFs)
│   ├── lab_guidelines.pdf         # Lab work instructions (131 pages)
│   ├── practical_guidelines.pdf   # Practical work instructions (27 pages)
│   └── research_task_topics.pdf   # Research topics list
│
├── labs/                          # Laboratory Works
│   ├── lab1_regression/           # Regression Analysis
│   ├── lab2_clustering/           # Clustering & Decision Trees
│   ├── lab3_apriori/              # Association Rules (Apriori)
│   └── lab4_genetic_algorithms/   # Genetic Algorithms
│
├── practicals/                    # Practical Works
│   ├── practical1_ols/            # OLS Regression Method
│   └── practical2_tfidf/          # TF-IDF Text Classification
│
├── research/                      # Research Task
│   ├── REPORT.md                  # Full research report
│   └── rnn_lstm_comparison.py     # RNN vs LSTM comparison
│
├── venv/                          # Python virtual environment
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone/navigate to project
cd "Data Analysis"

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Assignments Overview

### Laboratory Works

| Lab | Topic | Key Methods |
|-----|-------|-------------|
| Lab 1 | Regression Analysis | Linear, Multiple, Polynomial, Logistic Regression |
| Lab 2 | Clustering & Classification | K-Means, Hierarchical, DBSCAN, Decision Trees |
| Lab 3 | Association Rules | Apriori Algorithm, Support/Confidence/Lift |
| Lab 4 | Genetic Algorithms | Selection, Crossover, Mutation, Optimization |

### Practical Works

| Work | Topic | Key Methods |
|------|-------|-------------|
| Practical 1 | OLS Regression | Ordinary Least Squares, Statistical Analysis |
| Practical 2 | Text Classification | TF-IDF, Vector Space Model, Cosine Similarity |

### Research Task

**Topic #1**: Comparison of RNN and LSTM Neural Networks for NLP
- Theoretical analysis of architectures
- Vanishing gradient problem
- Experimental comparison
- Long-range dependency learning

---

## Running the Code

### Lab 1: Regression Analysis
```bash
source venv/bin/activate
cd labs/lab1_regression
python regression_analysis.py
```

### Lab 2: Clustering
```bash
cd labs/lab2_clustering
python clustering_analysis.py
```

### Lab 3: Apriori
```bash
cd labs/lab3_apriori
python apriori_analysis.py
```

### Lab 4: Genetic Algorithms
```bash
cd labs/lab4_genetic_algorithms
python genetic_algorithms.py
```

### Practical 1: OLS
```bash
cd practicals/practical1_ols
python ols_regression.py
```

### Practical 2: TF-IDF
```bash
cd practicals/practical2_tfidf
python tfidf_classification.py
```

### Research: RNN vs LSTM
```bash
cd research
python rnn_lstm_comparison.py
```

---

## Technologies Used

- **Python 3.12**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **Scikit-learn** - Machine Learning
- **Statsmodels** - Statistical modeling
- **MLxtend** - Association rules (Apriori)
- **PyTorch** - Deep Learning (RNN/LSTM)
- **SciPy** - Scientific computing

---

## License

Educational project for KhNURE coursework.

---

## Author

**Голодніков Дмитро**

Master's student, Software Engineering (ІПЗм-24-2)

KhNURE, 2025-2026
