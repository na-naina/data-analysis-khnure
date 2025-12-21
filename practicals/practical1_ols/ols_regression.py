"""
Practical Work #1: Regression Analysis - OLS Method
====================================================
Ordinary Least Squares (OLS) regression analysis

This module implements OLS regression analysis based on the practical work
guidelines using Python instead of STATISTICA.

Author: Student
Course: Intelligent Data Analysis (KhNURE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
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


def load_enterprise_data():
    """
    Load enterprise data from the practical work guidelines.
    Data characterizes production-economic activity of a machine-building enterprise.

    Variables:
    Y1 - Labor productivity (Продуктивність праці)
    Y2 - Cost reduction index (Індекс зниження собівартості)
    Y3 - Profitability (Рентабельність)
    X4 - Labor intensity per unit (Трудомісткість одиниці продукції)
    X5 - Share of workers in PPP (Питома вага робітників у складі ППП)
    X6 - Share of purchased products (Питома вага покупних виробів)
    X7 - Equipment shift coefficient (Коефіцієнт змінності обладнання)
    X8 - Bonuses per worker (Премії та винагороди на одного працівника)
    X9 - Share of defect losses (Питома вага втрат від шлюбу)
    X10 - Capital productivity (Фондовіддача)
    X11 - Average annual number of PPP (Середньорічна чисельність ППП)
    X12 - Average annual value of OPF (Середньорічна вартість ОПФ)
    X13 - Average annual wage fund (Середньорічний фонд заробітної плати ППП)
    X14 - Capital-labor ratio (Фондоозброєність праці)
    X15 - Turnover of standardized working capital (Оборотність нормованих оборотних коштів)
    X16 - Turnover of non-standardized working capital (Оборотність ненормованих оборотних засобів)
    X17 - Non-production costs (Невиробничі витрати)
    """

    # Data from Table 1.1 in the guidelines (first 30 enterprises)
    data = {
        'Y1': [9.26, 9.38, 12.11, 10.81, 9.35, 9.87, 8.17, 9.12, 5.88, 6.30,
               6.22, 5.49, 6.50, 6.61, 4.32, 7.37, 7.02, 8.25, 8.15, 8.72,
               6.64, 8.10, 5.52, 9.37, 13.17, 6.67, 5.68, 5.22, 10.02, 8.16],
        'Y2': [204.20, 209.60, 222.60, 236.70, 62.00, 53.10, 172.10, 56.50, 52.60, 46.60,
               53.20, 30.10, 146.40, 18.10, 13.60, 89.80, 62.50, 46.30, 103.50, 73.30,
               76.60, 73.01, 32.30, 199.60, 598.10, 71.20, 90.80, 82.10, 76.20, 119.50],
        'Y3': [13.26, 10.16, 13.72, 12.85, 10.63, 9.12, 25.83, 23.39, 14.68, 10.05,
               13.99, 9.68, 10.03, 9.13, 5.37, 9.86, 12.62, 5.02, 21.18, 25.17,
               19.40, 21.0, 6.57, 14.19, 15.81, 5.23, 7.99, 17.50, 17.16, 14.54],
        'X4': [0.23, 0.24, 0.19, 0.17, 0.23, 0.43, 0.31, 0.26, 0.49, 0.36,
               0.37, 0.43, 0.35, 0.38, 0.42, 0.30, 0.32, 0.25, 0.31, 0.26,
               0.37, 0.29, 0.34, 0.23, 0.17, 0.29, 0.41, 0.41, 0.22, 0.29],
        'X5': [0.78, 0.75, 0.68, 0.70, 0.62, 0.76, 0.73, 0.71, 0.69, 0.73,
               0.68, 0.74, 0.66, 0.72, 0.68, 0.77, 0.78, 0.78, 0.81, 0.79,
               0.77, 0.78, 0.72, 0.79, 0.77, 0.80, 0.71, 0.79, 0.76, 0.78],
        'X6': [0.40, 0.26, 0.40, 0.50, 0.40, 0.19, 0.25, 0.44, 0.17, 0.39,
               0.33, 0.25, 0.32, 0.02, 0.06, 0.15, 0.08, 0.20, 0.20, 0.30,
               0.24, 0.10, 0.11, 0.47, 0.53, 0.34, 0.20, 0.24, 0.54, 0.40],
        'X7': [1.37, 1.49, 1.44, 1.42, 1.35, 1.39, 1.16, 1.27, 1.16, 1.25,
               1.13, 1.10, 1.15, 1.23, 1.39, 1.38, 1.35, 1.42, 1.37, 1.41,
               1.35, 1.48, 1.24, 1.40, 1.45, 1.40, 1.28, 1.33, 1.22, 1.28],
        'X8': [1.23, 1.04, 1.80, 0.43, 0.88, 0.57, 1.72, 1.70, 0.84, 0.60,
               0.82, 0.84, 0.67, 1.04, 0.66, 0.86, 0.79, 0.34, 1.60, 1.46,
               1.27, 1.58, 0.68, 0.86, 1.98, 0.33, 0.45, 0.74, 0.03, 0.99],
        'X9': [0.23, 0.39, 0.43, 0.18, 0.15, 0.34, 0.38, 0.09, 0.14, 0.21,
               0.42, 0.05, 0.29, 0.48, 0.41, 0.62, 0.56, 1.76, 1.31, 0.45,
               0.50, 0.77, 1.20, 0.21, 0.25, 0.15, 0.66, 0.74, 0.32, 0.68],
        'X10': [1.45, 1.30, 1.37, 1.65, 1.91, 1.68, 1.94, 1.89, 1.94, 2.06,
                1.96, 1.02, 1.85, 0.88, 0.62, 1.09, 1.60, 1.53, 1.40, 2.22,
                1.32, 1.48, 0.68, 2.30, 1.37, 1.51, 1.43, 1.82, 2.62, None]
    }

    # Fill None with median
    df = pd.DataFrame(data)
    df['X10'] = df['X10'].fillna(df['X10'].median())

    return df


def descriptive_statistics(df):
    """Calculate and display descriptive statistics."""
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    stats_df = df.describe().T
    stats_df['skewness'] = df.skew()
    stats_df['kurtosis'] = df.kurtosis()

    print(stats_df.to_string())

    return stats_df


def correlation_analysis(df, target='Y1'):
    """Perform correlation analysis."""
    print("\n" + "=" * 60)
    print(f"CORRELATION ANALYSIS (with {target})")
    print("=" * 60)

    correlations = df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
    print("\nCorrelations with", target + ":")
    for var, corr in correlations.items():
        significance = "***" if abs(corr) > 0.5 else "**" if abs(corr) > 0.3 else "*" if abs(corr) > 0.1 else ""
        print(f"  {var}: {corr:+.4f} {significance}")

    return correlations


def plot_correlation_matrix(df, save_path=None):
    """Plot correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))

    corr_matrix = df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5
    )

    ax.set_title('Correlation Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def ols_regression(X, y, feature_names=None):
    """
    Perform OLS regression using statsmodels for detailed statistics.

    Parameters:
    -----------
    X : array-like
        Independent variables
    y : array-like
        Dependent variable
    feature_names : list
        Names of features

    Returns:
    --------
    OLSResults : statsmodels regression results
    """
    X = np.array(X)
    y = np.array(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Add constant for intercept
    X_const = sm.add_constant(X)

    # Fit OLS model
    model = sm.OLS(y, X_const)
    results = model.fit()

    return results


def plot_regression_diagnostics(results, X, y, feature_name='X', target_name='Y',
                                save_path=None):
    """Plot regression diagnostic plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    y_pred = results.fittedvalues
    residuals = results.resid
    standardized_residuals = results.get_influence().resid_studentized_internal

    # 1. Fitted vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, y, alpha=0.7, s=50)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Actual Values')
    ax1.set_title('Predicted vs Actual')
    ax1.grid(True, alpha=0.3)

    # 2. Residuals vs Fitted
    ax2 = axes[0, 1]
    ax2.scatter(y_pred, residuals, alpha=0.7, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Fitted (Homoscedasticity Check)')
    ax2.grid(True, alpha=0.3)

    # 3. Q-Q Plot (Normality)
    ax3 = axes[1, 0]
    stats.probplot(standardized_residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normality Check)')
    ax3.grid(True, alpha=0.3)

    # 4. Histogram of Residuals
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins='auto', density=True, edgecolor='black', alpha=0.7)
    xmin, xmax = ax4.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, residuals.mean(), residuals.std())
    ax4.plot(x, p, 'r-', linewidth=2, label='Normal fit')
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Density')
    ax4.set_title('Residual Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def stepwise_selection(X, y, feature_names, threshold_in=0.05, threshold_out=0.1):
    """
    Perform stepwise regression for feature selection.

    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target
    feature_names : list
        Names of features
    threshold_in : float
        p-value threshold for adding features
    threshold_out : float
        p-value threshold for removing features

    Returns:
    --------
    list : Selected feature names
    """
    included = []
    feature_names = list(feature_names)

    while True:
        changed = False

        # Forward step
        excluded = list(set(feature_names) - set(included))
        best_pvalue = float('inf')
        best_feature = None

        for feature in excluded:
            features = included + [feature]
            X_subset = X[features]
            X_const = sm.add_constant(X_subset)
            model = sm.OLS(y, X_const).fit()
            pvalue = model.pvalues.iloc[-1]

            if pvalue < best_pvalue:
                best_pvalue = pvalue
                best_feature = feature

        if best_pvalue < threshold_in:
            included.append(best_feature)
            changed = True
            print(f"Add: {best_feature}, p-value: {best_pvalue:.4f}")

        # Backward step
        if included:
            X_subset = X[included]
            X_const = sm.add_constant(X_subset)
            model = sm.OLS(y, X_const).fit()
            pvalues = model.pvalues.iloc[1:]  # Exclude constant
            worst_pvalue = pvalues.max()

            if worst_pvalue > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                print(f"Remove: {worst_feature}, p-value: {worst_pvalue:.4f}")

        if not changed:
            break

    return included


def multicollinearity_check(X, feature_names):
    """
    Check for multicollinearity using VIF (Variance Inflation Factor).

    Parameters:
    -----------
    X : DataFrame
        Features
    feature_names : list
        Names of features

    Returns:
    --------
    DataFrame : VIF values
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_array = np.array(X[feature_names])
    X_with_const = sm.add_constant(X_array)

    vif_data = pd.DataFrame()
    vif_data['Feature'] = feature_names
    vif_data['VIF'] = [variance_inflation_factor(X_with_const, i+1) for i in range(len(feature_names))]

    return vif_data.sort_values('VIF', ascending=False)


def run_full_analysis():
    """Run complete OLS regression analysis."""
    print("=" * 60)
    print("PRACTICAL WORK #1: OLS REGRESSION ANALYSIS")
    print("=" * 60)

    # Load data
    df = load_enterprise_data()
    print(f"\nLoaded data: {df.shape[0]} enterprises, {df.shape[1]} variables")

    # Descriptive statistics
    desc_stats = descriptive_statistics(df)

    # Correlation analysis
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    corr_y1 = correlation_analysis(df, 'Y1')
    plot_correlation_matrix(df, save_path=os.path.join(RESULTS_DIR, 'correlation_matrix.png'))

    # Select predictors for Y1 (Labor productivity)
    predictors = ['X4', 'X5', 'X7', 'X10']  # Based on correlations
    target = 'Y1'

    print("\n" + "=" * 60)
    print(f"OLS REGRESSION: {target} ~ {' + '.join(predictors)}")
    print("=" * 60)

    X = df[predictors]
    y = df[target]

    # Fit OLS model
    results = ols_regression(X, y, predictors)

    print("\nOLS Regression Summary:")
    print(results.summary())

    # Diagnostic plots
    plot_regression_diagnostics(results, X, y, save_path=os.path.join(RESULTS_DIR, 'regression_diagnostics.png'))

    # VIF analysis for multicollinearity
    print("\n" + "=" * 60)
    print("MULTICOLLINEARITY CHECK (VIF)")
    print("=" * 60)

    vif = multicollinearity_check(df, predictors)
    print(vif.to_string(index=False))
    print("\nInterpretation: VIF > 5 indicates potential multicollinearity")

    # Stepwise selection
    print("\n" + "=" * 60)
    print("STEPWISE FEATURE SELECTION")
    print("=" * 60)

    all_features = ['X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
    selected = stepwise_selection(df, df['Y1'], all_features)
    print(f"\nSelected features: {selected}")

    # Final model with selected features
    if selected:
        print("\n" + "=" * 60)
        print("FINAL MODEL WITH SELECTED FEATURES")
        print("=" * 60)

        final_results = ols_regression(df[selected], df['Y1'], selected)
        print(final_results.summary())

    # Model interpretation
    print("\n" + "=" * 60)
    print("MODEL INTERPRETATION")
    print("=" * 60)

    print(f"\nR-squared: {results.rsquared:.4f}")
    print(f"Adjusted R-squared: {results.rsquared_adj:.4f}")
    print(f"F-statistic: {results.fvalue:.4f}")
    print(f"F p-value: {results.f_pvalue:.4e}")

    print("\nCoefficients:")
    for i, (name, coef) in enumerate(zip(['Intercept'] + predictors, results.params)):
        pval = results.pvalues[i]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"  {name}: {coef:.4f} (p={pval:.4f}) {sig}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_analysis()
