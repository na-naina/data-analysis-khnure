# Lab Work #4: Genetic Algorithms

## Objective
Learn genetic algorithm optimization techniques:
- Selection methods (tournament, roulette, rank)
- Crossover operators (single-point, two-point, uniform)
- Mutation operators
- Function optimization

## Theoretical Background

### Genetic Algorithm Concepts
Genetic algorithms are inspired by biological evolution:
- **Population**: Set of candidate solutions
- **Chromosome**: Encoded solution (array of genes)
- **Fitness**: Quality measure of a solution
- **Selection**: Choosing parents based on fitness
- **Crossover**: Combining parent chromosomes
- **Mutation**: Random changes to chromosomes

### Algorithm Steps
1. Initialize random population
2. Evaluate fitness of all individuals
3. Select parents based on fitness
4. Apply crossover to create offspring
5. Apply mutation to offspring
6. Replace population (with elitism)
7. Repeat until termination condition

### Selection Methods
- **Tournament**: Select best from random subset
- **Roulette Wheel**: Probability proportional to fitness
- **Rank-based**: Probability based on rank

### Crossover Methods
- **Single-point**: Split at one random point
- **Two-point**: Split at two random points
- **Uniform**: Random gene-by-gene selection

### Test Functions
| Function | Global Minimum | Challenge |
|----------|---------------|-----------|
| Sphere | x=[0,...,0], f=0 | Simple, unimodal |
| Rastrigin | x=[0,...,0], f=0 | Many local minima |
| Rosenbrock | x=[1,...,1], f=0 | Narrow valley |
| Ackley | x=[0,...,0], f=0 | Flat regions |

## Implementation

### Files
- `genetic_algorithms.py` - Complete GA implementation

### Usage
```bash
source venv/bin/activate
cd labs/lab4_genetic_algorithms
python genetic_algorithms.py
```

### Key Parameters
```python
GeneticAlgorithm(
    fitness_func=sphere_function,
    n_genes=2,
    gene_bounds=(-5, 5),
    population_size=100,
    n_generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elitism=2,
    selection_method='tournament'
)
```

## Tasks Completed

1. ✅ Generic GA class implementation
2. ✅ Tournament, roulette, and rank selection
3. ✅ Single-point, two-point, uniform crossover
4. ✅ Gaussian mutation with bounds
5. ✅ Elitism preservation
6. ✅ Sphere function optimization
7. ✅ Rastrigin function (multimodal)
8. ✅ Rosenbrock function (valley)
9. ✅ Selection method comparison
10. ✅ High-dimensional optimization (10D)

## Output Files
- `sphere_evolution.png` - Sphere optimization progress
- `sphere_surface.png` - 3D surface visualization
- `rastrigin_evolution.png` - Rastrigin optimization
- `rastrigin_surface.png` - Multimodal landscape
- `rosenbrock_evolution.png` - Rosenbrock optimization
- `selection_comparison.png` - Method comparison boxplot

## Dependencies
- numpy
- matplotlib
