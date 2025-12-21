# Practical Work #1: OLS Regression Analysis

## Objective
Solve regression problems using the Ordinary Least Squares (OLS) method.

## Data Description
Enterprise production-economic activity data with the following variables:

| Variable | Description |
|----------|-------------|
| Y1 | Labor productivity |
| Y2 | Cost reduction index |
| Y3 | Profitability |
| X4 | Labor intensity per unit |
| X5 | Share of workers in PPP |
| X6 | Share of purchased products |
| X7 | Equipment shift coefficient |
| X8 | Bonuses per worker |
| X9 | Share of defect losses |
| X10 | Capital productivity |

## Implementation

### Files
- `ols_regression.py` - Complete OLS analysis implementation

### Usage
```bash
source venv/bin/activate
cd practicals/practical1_ols
python ols_regression.py
```

## Tasks Completed

1. ✅ Descriptive statistics analysis
2. ✅ Correlation matrix visualization
3. ✅ OLS regression with statsmodels
4. ✅ Regression diagnostics (Q-Q plot, residuals)
5. ✅ Multicollinearity check (VIF)
6. ✅ Stepwise feature selection
7. ✅ Model interpretation

## Output Files
- `correlation_matrix.png` - Correlation heatmap
- `regression_diagnostics.png` - Diagnostic plots

## Key Results

### Regression Equation
```
Y1 = β₀ + β₁*X4 + β₂*X5 + β₃*X7 + β₄*X10
```

### Model Quality Metrics
- R² - Coefficient of determination
- Adjusted R² - Adjusted for number of predictors
- F-statistic - Overall model significance
- p-values - Individual coefficient significance

## Dependencies
- numpy, pandas
- statsmodels
- scipy
- matplotlib, seaborn
