"""
Lab Work #1: Regression Analysis
================================
Simple, Multiple, and Non-linear Regression Analysis

This module implements regression analysis techniques covered in the lab:
- Simple Linear Regression
- Multiple Linear Regression
- Non-linear Regression (Polynomial, Logistic)

Author: Student
Course: Intelligent Data Analysis (KhNURE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set style for plots - professional academic style
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
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
sns.set_palette(COLORS)

# Results directory
import os
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_sample_data():
    """Generate sample data similar to STATISTICA example."""
    np.random.seed(42)
    n = 11

    # VAR1: sequential values 0-10
    var1 = np.arange(n)

    # VAR2: random values 0-1
    var2 = np.random.rand(n)

    # VAR3: sum of VAR1 + VAR2 (with some noise)
    var3 = var1 + var2

    return pd.DataFrame({
        'VAR1': var1,
        'VAR2': var2,
        'VAR3': var3
    })


def simple_linear_regression(X, y, feature_name='X', target_name='Y'):
    """
    Perform simple linear regression analysis.

    Parameters:
    -----------
    X : array-like
        Independent variable
    y : array-like
        Dependent variable
    feature_name : str
        Name of independent variable
    target_name : str
        Name of dependent variable

    Returns:
    --------
    dict : Results containing coefficients, metrics, and model
    """
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)

    # Statistical analysis using statsmodels
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()

    results = {
        'intercept': model.intercept_,
        'coefficient': model.coef_[0],
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'equation': f"{target_name} = {model.intercept_:.4f} + {model.coef_[0]:.4f} * {feature_name}",
        'model': model,
        'summary': model_sm.summary(),
        'p_values': model_sm.pvalues,
        'residuals': y - y_pred
    }

    return results


def plot_simple_regression(X, y, results, feature_name='X', target_name='Y',
                          title='Simple Linear Regression', save_path=None):
    """Plot simple linear regression with regression line."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    X = np.array(X)
    y = np.array(y)
    y_pred = results['model'].predict(X.reshape(-1, 1))

    # 1. Scatter plot with regression line
    ax1 = axes[0, 0]
    ax1.scatter(X, y, alpha=0.7, label='Data points', s=50)
    ax1.plot(X, y_pred, 'r-', linewidth=2, label='Regression line')
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel(target_name)
    ax1.set_title(f'{title}\n{results["equation"]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals plot
    ax2 = axes[0, 1]
    residuals = results['residuals']
    ax2.scatter(y_pred, residuals, alpha=0.7, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted')
    ax2.grid(True, alpha=0.3)

    # 3. Q-Q plot for residuals (normality check)
    ax3 = axes[1, 0]
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot of Residuals')
    ax3.grid(True, alpha=0.3)

    # 4. Histogram of residuals
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins='auto', edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Residuals')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def multiple_linear_regression(X, y, feature_names=None, target_name='Y'):
    """
    Perform multiple linear regression analysis.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Independent variables
    y : array-like
        Dependent variable
    feature_names : list
        Names of independent variables
    target_name : str
        Name of dependent variable

    Returns:
    --------
    dict : Results containing coefficients, metrics, and model
    """
    X = np.array(X)
    y = np.array(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f'X{i+1}' for i in range(n_features)]

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Calculate metrics
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_features - 1)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # Statistical analysis using statsmodels
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()

    # Build equation string
    equation_parts = [f"{model.intercept_:.4f}"]
    for i, (coef, name) in enumerate(zip(model.coef_, feature_names)):
        equation_parts.append(f"{coef:+.4f}*{name}")
    equation = f"{target_name} = " + " ".join(equation_parts)

    results = {
        'intercept': model.intercept_,
        'coefficients': dict(zip(feature_names, model.coef_)),
        'r2': r2,
        'adj_r2': adj_r2,
        'mse': mse,
        'rmse': rmse,
        'equation': equation,
        'model': model,
        'summary': model_sm.summary(),
        'p_values': model_sm.pvalues,
        'residuals': y - y_pred,
        'feature_names': feature_names
    }

    return results


def multivariate_regression_example():
    """
    Example from the lab: Y = B0 + B1*X + B2*Z

    Y - crop yield (врожайність)
    X - fertilizer (добрива)
    Z - precipitation (опади)
    """
    # Data from the lab manual
    data = pd.DataFrame({
        'Y': [40, 50, 50, 70, 65, 65, 80],  # Crop yield
        'X': [100, 200, 300, 400, 500, 600, 700],  # Fertilizer
        'Z': [10, 20, 10, 30, 20, 20, 30]  # Precipitation
    })

    X = data[['X', 'Z']].values
    y = data['Y'].values

    results = multiple_linear_regression(X, y, feature_names=['X', 'Z'], target_name='Y')

    return data, results


def predict_with_confidence(model_results, X_new, confidence=0.95):
    """
    Predict values with confidence intervals.

    Parameters:
    -----------
    model_results : dict
        Results from regression analysis
    X_new : array-like
        New data points for prediction
    confidence : float
        Confidence level (default 0.95)

    Returns:
    --------
    dict : Predictions with confidence intervals
    """
    model = model_results['model']
    X_new = np.array(X_new)

    if X_new.ndim == 1:
        X_new = X_new.reshape(-1, 1)

    y_pred = model.predict(X_new)

    # For proper confidence intervals, we would need the full statsmodels results
    # This is a simplified version
    n = len(model_results['residuals'])
    se = np.std(model_results['residuals'])
    t_value = stats.t.ppf((1 + confidence) / 2, n - 2)
    margin = t_value * se

    return {
        'prediction': y_pred,
        'lower_bound': y_pred - margin,
        'upper_bound': y_pred + margin,
        'confidence_level': confidence
    }


def polynomial_regression(X, y, degree=2, feature_name='X', target_name='Y'):
    """
    Perform polynomial regression.

    Parameters:
    -----------
    X : array-like
        Independent variable
    y : array-like
        Dependent variable
    degree : int
        Polynomial degree

    Returns:
    --------
    dict : Results containing coefficients, metrics, and model
    """
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Fit model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predictions
    y_pred = model.predict(X_poly)

    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    results = {
        'intercept': model.intercept_,
        'coefficients': model.coef_,
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'degree': degree,
        'model': model,
        'poly_transformer': poly,
        'residuals': y - y_pred
    }

    return results


def plot_polynomial_regression(X, y, results, feature_name='X', target_name='Y',
                               save_path=None):
    """Plot polynomial regression."""
    fig, ax = plt.subplots(figsize=(10, 6))

    X = np.array(X)
    y = np.array(y)

    # Scatter plot
    ax.scatter(X, y, alpha=0.7, label='Data points', s=50)

    # Generate smooth curve for plotting
    X_smooth = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_smooth_poly = results['poly_transformer'].transform(X_smooth)
    y_smooth = results['model'].predict(X_smooth_poly)

    ax.plot(X_smooth, y_smooth, 'r-', linewidth=2,
            label=f'Polynomial (degree={results["degree"]})')

    ax.set_xlabel(feature_name)
    ax.set_ylabel(target_name)
    ax.set_title(f'Polynomial Regression (R² = {results["r2"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def logistic_regression_example():
    """
    Logistic regression example for binary classification.
    """
    np.random.seed(42)

    # Generate sample data
    n = 100
    X = np.random.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Add some noise
    noise_idx = np.random.choice(n, size=10, replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    # Fit logistic regression
    model = LogisticRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Accuracy
    accuracy = (y_pred == y).mean()

    return {
        'model': model,
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy,
        'coefficients': model.coef_,
        'intercept': model.intercept_
    }


def detect_outliers(residuals, threshold=2.0):
    """
    Detect outliers based on standardized residuals.

    Parameters:
    -----------
    residuals : array-like
        Model residuals
    threshold : float
        Z-score threshold for outlier detection

    Returns:
    --------
    array : Boolean mask indicating outliers
    """
    z_scores = np.abs(stats.zscore(residuals))
    return z_scores > threshold


def run_full_analysis():
    """Run complete regression analysis demonstration."""
    print("=" * 60)
    print("LAB 1: REGRESSION ANALYSIS")
    print("=" * 60)

    # Part 1: Simple Linear Regression
    print("\n" + "=" * 60)
    print("PART 1: Simple Linear Regression")
    print("=" * 60)

    data = generate_sample_data()
    print("\nSample Data:")
    print(data.to_string(index=False))

    results = simple_linear_regression(
        data['VAR1'], data['VAR3'],
        feature_name='VAR1', target_name='VAR3'
    )

    print(f"\nRegression Equation: {results['equation']}")
    print(f"R² Score: {results['r2']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")

    plot_simple_regression(
        data['VAR1'], data['VAR3'], results,
        feature_name='VAR1', target_name='VAR3',
        title='Simple Linear Regression Example',
        save_path=os.path.join(RESULTS_DIR, 'simple_regression.png')
    )

    # Part 2: Multiple Linear Regression
    print("\n" + "=" * 60)
    print("PART 2: Multiple Linear Regression")
    print("=" * 60)

    crop_data, crop_results = multivariate_regression_example()
    print("\nCrop Yield Data:")
    print(crop_data.to_string(index=False))

    print(f"\nRegression Equation: {crop_results['equation']}")
    print(f"R² Score: {crop_results['r2']:.4f}")
    print(f"Adjusted R²: {crop_results['adj_r2']:.4f}")

    print("\nCoefficients:")
    for name, coef in crop_results['coefficients'].items():
        print(f"  {name}: {coef:.4f}")

    print("\n" + "-" * 40)
    print("Statsmodels Summary:")
    print(crop_results['summary'])

    # Prediction example
    X_new = [[350, 25]]  # Fertilizer=350, Precipitation=25
    prediction = predict_with_confidence(crop_results, X_new)
    print(f"\nPrediction for X=350, Z=25:")
    print(f"  Predicted Y: {prediction['prediction'][0]:.2f}")
    print(f"  95% CI: [{prediction['lower_bound'][0]:.2f}, {prediction['upper_bound'][0]:.2f}]")

    # Part 3: Polynomial Regression
    print("\n" + "=" * 60)
    print("PART 3: Polynomial Regression")
    print("=" * 60)

    np.random.seed(42)
    X_poly = np.linspace(0, 10, 50)
    y_poly = 2 + 3*X_poly - 0.5*X_poly**2 + np.random.randn(50) * 2

    poly_results = polynomial_regression(X_poly, y_poly, degree=2)
    print(f"\nPolynomial Degree: {poly_results['degree']}")
    print(f"R² Score: {poly_results['r2']:.4f}")
    print(f"Coefficients: {poly_results['coefficients']}")

    plot_polynomial_regression(
        X_poly, y_poly, poly_results,
        save_path=os.path.join(RESULTS_DIR, 'polynomial_regression.png')
    )

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_analysis()
