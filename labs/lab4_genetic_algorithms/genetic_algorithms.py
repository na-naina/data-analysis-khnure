"""
Lab Work #4: Genetic Algorithms
================================
Optimization using Genetic Algorithms

This module implements:
- Basic Genetic Algorithm
- Various selection methods (roulette, tournament)
- Crossover operators (single-point, two-point, uniform)
- Mutation operators
- Function optimization examples

Author: Student
Course: Intelligent Data Analysis (KhNURE)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional
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

# Results directory
import os
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


class GeneticAlgorithm:
    """
    Generic Genetic Algorithm implementation for optimization problems.
    """

    def __init__(
        self,
        fitness_func: Callable,
        n_genes: int,
        gene_bounds: Tuple[float, float] = (-10, 10),
        population_size: int = 100,
        n_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism: int = 2,
        selection_method: str = 'tournament',
        crossover_method: str = 'single_point',
        maximize: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize Genetic Algorithm.

        Parameters:
        -----------
        fitness_func : Callable
            Function to optimize (takes array of genes, returns fitness)
        n_genes : int
            Number of genes (dimensions)
        gene_bounds : tuple
            (min, max) bounds for gene values
        population_size : int
            Number of individuals in population
        n_generations : int
            Number of generations to evolve
        crossover_rate : float
            Probability of crossover (0-1)
        mutation_rate : float
            Probability of mutation per gene (0-1)
        elitism : int
            Number of best individuals to preserve
        selection_method : str
            'tournament', 'roulette', or 'rank'
        crossover_method : str
            'single_point', 'two_point', or 'uniform'
        maximize : bool
            True for maximization, False for minimization
        random_state : int
            Random seed for reproducibility
        """
        self.fitness_func = fitness_func
        self.n_genes = n_genes
        self.gene_bounds = gene_bounds
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.maximize = maximize

        if random_state is not None:
            np.random.seed(random_state)

        # Initialize population
        self.population = self._initialize_population()
        self.fitness_history = []
        self.best_history = []
        self.avg_history = []

    def _initialize_population(self) -> np.ndarray:
        """Initialize random population within bounds."""
        low, high = self.gene_bounds
        return np.random.uniform(low, high, (self.population_size, self.n_genes))

    def _evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness for all individuals."""
        fitness = np.array([self.fitness_func(ind) for ind in population])
        if not self.maximize:
            # Convert minimization to maximization
            fitness = -fitness
        return fitness

    def _selection_tournament(self, fitness: np.ndarray, tournament_size: int = 3) -> int:
        """Tournament selection."""
        candidates = np.random.choice(len(fitness), tournament_size, replace=False)
        winner = candidates[np.argmax(fitness[candidates])]
        return winner

    def _selection_roulette(self, fitness: np.ndarray) -> int:
        """Roulette wheel selection."""
        # Shift fitness to positive values
        shifted = fitness - fitness.min() + 1e-6
        probabilities = shifted / shifted.sum()
        return np.random.choice(len(fitness), p=probabilities)

    def _selection_rank(self, fitness: np.ndarray) -> int:
        """Rank-based selection."""
        ranks = np.argsort(np.argsort(fitness)) + 1
        probabilities = ranks / ranks.sum()
        return np.random.choice(len(fitness), p=probabilities)

    def _select_parents(self, fitness: np.ndarray) -> Tuple[int, int]:
        """Select two parents using chosen selection method."""
        if self.selection_method == 'tournament':
            p1 = self._selection_tournament(fitness)
            p2 = self._selection_tournament(fitness)
        elif self.selection_method == 'roulette':
            p1 = self._selection_roulette(fitness)
            p2 = self._selection_roulette(fitness)
        else:  # rank
            p1 = self._selection_rank(fitness)
            p2 = self._selection_rank(fitness)

        return p1, p2

    def _crossover_single_point(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover."""
        point = np.random.randint(1, self.n_genes)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _crossover_two_point(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Two-point crossover."""
        points = sorted(np.random.choice(self.n_genes, 2, replace=False))
        p1, p2 = points

        child1 = np.concatenate([parent1[:p1], parent2[p1:p2], parent1[p2:]])
        child2 = np.concatenate([parent2[:p1], parent1[p1:p2], parent2[p2:]])
        return child1, child2

    def _crossover_uniform(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover."""
        mask = np.random.random(self.n_genes) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply crossover with probability."""
        if np.random.random() < self.crossover_rate:
            if self.crossover_method == 'single_point':
                return self._crossover_single_point(parent1, parent2)
            elif self.crossover_method == 'two_point':
                return self._crossover_two_point(parent1, parent2)
            else:  # uniform
                return self._crossover_uniform(parent1, parent2)
        return parent1.copy(), parent2.copy()

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Apply mutation to individual."""
        low, high = self.gene_bounds
        for i in range(self.n_genes):
            if np.random.random() < self.mutation_rate:
                # Gaussian mutation
                individual[i] += np.random.normal(0, (high - low) * 0.1)
                # Ensure bounds
                individual[i] = np.clip(individual[i], low, high)
        return individual

    def evolve(self) -> Tuple[np.ndarray, float]:
        """
        Run the genetic algorithm evolution.

        Returns:
        --------
        tuple : (best_individual, best_fitness)
        """
        for generation in range(self.n_generations):
            # Evaluate fitness
            fitness = self._evaluate_fitness(self.population)

            # Track statistics
            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx] if self.maximize else -fitness[best_idx]
            avg_fitness = fitness.mean() if self.maximize else -fitness.mean()

            self.best_history.append(best_fitness)
            self.avg_history.append(avg_fitness)
            self.fitness_history.append(fitness.copy())

            # Elitism: preserve best individuals
            elite_indices = np.argsort(fitness)[-self.elitism:]
            elite = self.population[elite_indices].copy()

            # Create new population
            new_population = []

            while len(new_population) < self.population_size - self.elitism:
                # Selection
                p1_idx, p2_idx = self._select_parents(fitness)
                parent1 = self.population[p1_idx]
                parent2 = self.population[p2_idx]

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            # Trim to population size and add elite
            new_population = np.array(new_population[:self.population_size - self.elitism])
            self.population = np.vstack([new_population, elite])

        # Final evaluation
        final_fitness = self._evaluate_fitness(self.population)
        best_idx = np.argmax(final_fitness)

        best_individual = self.population[best_idx]
        best_fitness = final_fitness[best_idx] if self.maximize else -final_fitness[best_idx]

        return best_individual, best_fitness

    def plot_evolution(self, save_path: Optional[str] = None):
        """Plot evolution history."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        generations = range(len(self.best_history))

        # Best and average fitness
        ax1 = axes[0]
        ax1.plot(generations, self.best_history, 'b-', label='Best', linewidth=2)
        ax1.plot(generations, self.avg_history, 'r--', label='Average', linewidth=1)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Fitness diversity (std)
        ax2 = axes[1]
        std_history = [f.std() for f in self.fitness_history]
        ax2.plot(generations, std_history, 'g-', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness Std Dev')
        ax2.set_title('Population Diversity')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

        return fig


# ==================== TEST FUNCTIONS ====================

def sphere_function(x: np.ndarray) -> float:
    """
    Sphere function (minimization).
    Global minimum at x = [0, 0, ..., 0], f(x) = 0
    """
    return -np.sum(x ** 2)  # Negative for maximization


def rastrigin_function(x: np.ndarray) -> float:
    """
    Rastrigin function (minimization).
    Global minimum at x = [0, 0, ..., 0], f(x) = 0
    Many local minima.
    """
    A = 10
    n = len(x)
    return -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock function (minimization).
    Global minimum at x = [1, 1, ..., 1], f(x) = 0
    """
    return -np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def ackley_function(x: np.ndarray) -> float:
    """
    Ackley function (minimization).
    Global minimum at x = [0, 0, ..., 0], f(x) = 0
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -(-20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e)


def custom_function(x: np.ndarray) -> float:
    """
    Custom function: f(x) = sin(x1) * cos(x2) + x1^2 - x2^2
    """
    if len(x) >= 2:
        return np.sin(x[0]) * np.cos(x[1]) + x[0]**2 - x[1]**2
    return x[0]**2


# ==================== VISUALIZATION ====================

def plot_2d_function(func: Callable, bounds: Tuple[float, float] = (-5, 5),
                    resolution: int = 100, title: str = "Function",
                    best_point: Optional[np.ndarray] = None,
                    save_path: Optional[str] = None):
    """Plot 2D function surface and contour."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate function
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = -func(np.array([X[i, j], Y[i, j]]))  # Negate back

    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('f(x)')
    ax1.set_title(f'{title} - Surface')

    if best_point is not None:
        best_z = -func(best_point)
        ax1.scatter([best_point[0]], [best_point[1]], [best_z],
                   color='red', s=100, marker='*', label='Best')

    # Contour plot
    ax2 = axes[1]
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax2, label='f(x)')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_title(f'{title} - Contour')

    if best_point is not None:
        ax2.scatter([best_point[0]], [best_point[1]],
                   color='red', s=100, marker='*', label='Best', zorder=5)
        ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def compare_selection_methods(fitness_func: Callable, n_genes: int = 2,
                             gene_bounds: Tuple[float, float] = (-5, 5),
                             n_runs: int = 5, save_path: Optional[str] = None):
    """Compare different selection methods."""
    methods = ['tournament', 'roulette', 'rank']
    results = {method: [] for method in methods}

    for method in methods:
        for _ in range(n_runs):
            ga = GeneticAlgorithm(
                fitness_func=fitness_func,
                n_genes=n_genes,
                gene_bounds=gene_bounds,
                population_size=50,
                n_generations=50,
                selection_method=method,
                random_state=None
            )
            _, best_fitness = ga.evolve()
            results[method].append(best_fitness)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = range(len(methods))
    box_data = [results[method] for method in methods]

    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)

    colors = ['lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticklabels(methods)
    ax.set_xlabel('Selection Method')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Comparison of Selection Methods')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return results


def run_full_analysis():
    """Run complete Genetic Algorithm demonstration."""
    print("=" * 60)
    print("LAB 4: GENETIC ALGORITHMS")
    print("=" * 60)

    # Part 1: Sphere Function (Simple)
    print("\n" + "=" * 60)
    print("PART 1: Sphere Function Optimization")
    print("=" * 60)

    print("\nSphere Function: f(x) = Σ(xi²)")
    print("Global minimum at x = [0, 0], f(x) = 0")

    ga_sphere = GeneticAlgorithm(
        fitness_func=sphere_function,
        n_genes=2,
        gene_bounds=(-5, 5),
        population_size=100,
        n_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        selection_method='tournament',
        maximize=True,  # We negated the function
        random_state=42
    )

    best_sphere, best_fitness_sphere = ga_sphere.evolve()
    print(f"\nBest solution found: {best_sphere}")
    print(f"Best fitness (minimized): {-best_fitness_sphere:.6f}")

    ga_sphere.plot_evolution(save_path=os.path.join(RESULTS_DIR, 'sphere_evolution.png'))
    plot_2d_function(sphere_function, bounds=(-5, 5), title="Sphere Function",
                    best_point=best_sphere, save_path=os.path.join(RESULTS_DIR, 'sphere_surface.png'))

    # Part 2: Rastrigin Function (Complex, many local minima)
    print("\n" + "=" * 60)
    print("PART 2: Rastrigin Function Optimization")
    print("=" * 60)

    print("\nRastrigin Function: f(x) = An + Σ(xi² - A*cos(2πxi))")
    print("Global minimum at x = [0, 0], f(x) = 0")
    print("This function has many local minima!")

    ga_rastrigin = GeneticAlgorithm(
        fitness_func=rastrigin_function,
        n_genes=2,
        gene_bounds=(-5.12, 5.12),
        population_size=200,
        n_generations=150,
        crossover_rate=0.9,
        mutation_rate=0.15,
        selection_method='tournament',
        random_state=42
    )

    best_rastrigin, best_fitness_rastrigin = ga_rastrigin.evolve()
    print(f"\nBest solution found: {best_rastrigin}")
    print(f"Best fitness (minimized): {-best_fitness_rastrigin:.6f}")

    ga_rastrigin.plot_evolution(save_path=os.path.join(RESULTS_DIR, 'rastrigin_evolution.png'))
    plot_2d_function(rastrigin_function, bounds=(-5.12, 5.12),
                    title="Rastrigin Function", best_point=best_rastrigin,
                    save_path=os.path.join(RESULTS_DIR, 'rastrigin_surface.png'))

    # Part 3: Rosenbrock Function
    print("\n" + "=" * 60)
    print("PART 3: Rosenbrock Function Optimization")
    print("=" * 60)

    print("\nRosenbrock Function: f(x) = Σ[100(xi+1 - xi²)² + (1 - xi)²]")
    print("Global minimum at x = [1, 1], f(x) = 0")

    ga_rosenbrock = GeneticAlgorithm(
        fitness_func=rosenbrock_function,
        n_genes=2,
        gene_bounds=(-5, 5),
        population_size=150,
        n_generations=200,
        crossover_rate=0.9,
        mutation_rate=0.1,
        selection_method='tournament',
        random_state=42
    )

    best_rosenbrock, best_fitness_rosenbrock = ga_rosenbrock.evolve()
    print(f"\nBest solution found: {best_rosenbrock}")
    print(f"Best fitness (minimized): {-best_fitness_rosenbrock:.6f}")

    ga_rosenbrock.plot_evolution(save_path=os.path.join(RESULTS_DIR, 'rosenbrock_evolution.png'))
    plot_2d_function(rosenbrock_function, bounds=(-5, 5),
                    title="Rosenbrock Function", best_point=best_rosenbrock,
                    save_path=os.path.join(RESULTS_DIR, 'rosenbrock_surface.png'))

    # Part 4: Compare Selection Methods
    print("\n" + "=" * 60)
    print("PART 4: Comparing Selection Methods")
    print("=" * 60)

    comparison_results = compare_selection_methods(
        sphere_function, n_genes=2, gene_bounds=(-5, 5), n_runs=10,
        save_path=os.path.join(RESULTS_DIR, 'selection_comparison.png')
    )

    print("\nSelection Method Comparison (Best Fitness):")
    for method, values in comparison_results.items():
        print(f"  {method}: mean={np.mean(values):.6f}, std={np.std(values):.6f}")

    # Part 5: Higher Dimensions
    print("\n" + "=" * 60)
    print("PART 5: Higher Dimensional Optimization (10D)")
    print("=" * 60)

    ga_10d = GeneticAlgorithm(
        fitness_func=sphere_function,
        n_genes=10,
        gene_bounds=(-5, 5),
        population_size=200,
        n_generations=200,
        crossover_rate=0.9,
        mutation_rate=0.15,
        selection_method='tournament',
        random_state=42
    )

    best_10d, best_fitness_10d = ga_10d.evolve()
    print(f"\nBest solution found (10D): {best_10d}")
    print(f"Best fitness (minimized): {-best_fitness_10d:.6f}")
    print(f"Expected minimum: 0.0")

    ga_10d.plot_evolution(save_path=os.path.join(RESULTS_DIR, 'sphere_10d_evolution.png'))

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_analysis()
