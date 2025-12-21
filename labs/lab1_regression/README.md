# Lab Work #1: Regression Analysis

## Objective
Learn regression analysis techniques using Python (as an alternative to STATISTICA):
- Simple Linear Regression
- Multiple Linear Regression
- Non-linear Regression (Polynomial, Logistic)

## Theoretical Background

### Linear Regression
Linear regression models the relationship between a dependent variable Y and independent variable(s) X using a linear function:

**Simple:** `Y = β₀ + β₁X + ε`

**Multiple:** `Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε`

### Least Squares Method (OLS)
The method minimizes the sum of squared residuals:
```
min Σ(yᵢ - ŷᵢ)²
```

### Key Metrics
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model
- **Adjusted R²**: R² adjusted for the number of predictors
- **RMSE**: Root Mean Square Error
- **p-values**: Statistical significance of coefficients

## Implementation

### Files
- `regression_analysis.py` - Main implementation with all regression methods

### Usage
```bash
# Activate virtual environment
source venv/bin/activate

# Run analysis
cd labs/lab1_regression
python regression_analysis.py
```

### Example: Multiple Linear Regression
From the lab manual - predicting crop yield:

| Y (Yield) | X (Fertilizer) | Z (Precipitation) |
|-----------|----------------|-------------------|
| 40        | 100            | 10                |
| 50        | 200            | 20                |
| 50        | 300            | 10                |
| 70        | 400            | 30                |
| 65        | 500            | 20                |
| 65        | 600            | 20                |
| 80        | 700            | 30                |

**Result:** `Y = 28.095 + 0.038*X + 0.833*Z`

## Tasks Completed

1. ✅ Simple linear regression with visualization
2. ✅ Residual analysis and Q-Q plots
3. ✅ Multiple linear regression
4. ✅ Coefficient interpretation
5. ✅ Prediction with confidence intervals
6. ✅ Polynomial regression
7. ✅ Outlier detection

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn
- statsmodels
