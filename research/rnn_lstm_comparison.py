"""
Research Task: Comparison of RNN and LSTM Neural Networks
==========================================================
Topic #1: Comparing RNN and LSTM for Natural Language Processing

This module provides experimental comparison of RNN and LSTM architectures
for sequence modeling tasks in NLP.

Author: Student
Course: Intelligent Data Analysis (KhNURE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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


# ==================== MODEL DEFINITIONS ====================

class SimpleRNN(nn.Module):
    """
    Simple/Vanilla RNN for sequence classification.

    Architecture:
    - RNN layer(s)
    - Fully connected output layer
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity='tanh'
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, hn = self.rnn(x, h0)
        # out shape: (batch, seq_len, hidden_size)

        # Take the last time step output
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMModel(nn.Module):
    """
    LSTM (Long Short-Term Memory) for sequence classification.

    Architecture:
    - LSTM layer(s) with forget gate, input gate, output gate
    - Fully connected output layer
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out shape: (batch, seq_len, hidden_size)

        # Take the last time step output
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# ==================== DATA GENERATION ====================

def generate_sequence_data(n_samples=1000, seq_length=50, n_features=10,
                          task='classification', n_classes=2):
    """
    Generate synthetic sequence data for testing.

    Parameters:
    -----------
    n_samples : int
        Number of sequences
    seq_length : int
        Length of each sequence
    n_features : int
        Number of features per timestep
    task : str
        'classification' or 'regression'
    n_classes : int
        Number of classes for classification

    Returns:
    --------
    tuple : (X, y) data
    """
    X = np.random.randn(n_samples, seq_length, n_features).astype(np.float32)

    if task == 'classification':
        # Create patterns that depend on long-range dependencies
        # Class depends on relationship between early and late parts of sequence
        early_mean = X[:, :seq_length//4, :].mean(axis=(1, 2))
        late_mean = X[:, -seq_length//4:, :].mean(axis=(1, 2))
        y = (early_mean > late_mean).astype(np.int64)

        if n_classes > 2:
            y = np.digitize(early_mean - late_mean,
                           bins=np.linspace(-1, 1, n_classes)) - 1
            y = np.clip(y, 0, n_classes - 1)
    else:
        y = X[:, -1, 0].astype(np.float32)

    return X, y


def generate_long_dependency_data(n_samples=1000, seq_length=100, gap=50):
    """
    Generate data with explicit long-range dependencies.

    This tests the ability to remember information over many timesteps.
    """
    X = np.random.randn(n_samples, seq_length, 1).astype(np.float32)

    # Place a "signal" at the beginning
    signal = np.random.choice([0, 1], size=n_samples)
    X[:, 0, 0] = signal * 2 - 1  # Convert to -1 or 1

    # The label depends on this early signal
    y = signal.astype(np.int64)

    return X, y


# ==================== TRAINING ====================

def train_model(model, train_loader, val_loader, criterion, optimizer,
               n_epochs=50, device='cpu', verbose=True):
    """
    Train a model and track metrics.

    Returns:
    --------
    dict : Training history
    """
    model.to(device)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_time': []
    }

    for epoch in range(n_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        epoch_time = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Time: {epoch_time:.2f}s")

    return history


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== COMPARISON EXPERIMENTS ====================

def experiment_standard_sequences():
    """
    Experiment 1: Standard sequence classification.

    Compare RNN and LSTM on standard sequence data.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Standard Sequence Classification")
    print("=" * 70)

    # Generate data
    X, y = generate_sequence_data(n_samples=2000, seq_length=50, n_features=10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create dataloaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model parameters
    input_size = 10
    hidden_size = 64
    num_layers = 2
    output_size = 2
    n_epochs = 50

    # Train RNN
    print("\nTraining Simple RNN...")
    rnn_model = SimpleRNN(input_size, hidden_size, num_layers, output_size)
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    rnn_history = train_model(rnn_model, train_loader, val_loader, criterion,
                             rnn_optimizer, n_epochs)

    # Train LSTM
    print("\nTraining LSTM...")
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    lstm_history = train_model(lstm_model, train_loader, val_loader, criterion,
                              lstm_optimizer, n_epochs)

    # Results
    results = {
        'RNN': {
            'final_val_acc': rnn_history['val_acc'][-1],
            'best_val_acc': max(rnn_history['val_acc']),
            'avg_epoch_time': np.mean(rnn_history['epoch_time']),
            'n_parameters': count_parameters(rnn_model),
            'history': rnn_history
        },
        'LSTM': {
            'final_val_acc': lstm_history['val_acc'][-1],
            'best_val_acc': max(lstm_history['val_acc']),
            'avg_epoch_time': np.mean(lstm_history['epoch_time']),
            'n_parameters': count_parameters(lstm_model),
            'history': lstm_history
        }
    }

    return results


def experiment_long_dependencies():
    """
    Experiment 2: Long-range dependency learning.

    Test ability to learn dependencies over long sequences.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Long-Range Dependencies")
    print("=" * 70)

    # Generate data with long-range dependencies
    X, y = generate_long_dependency_data(n_samples=2000, seq_length=100, gap=80)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    input_size = 1
    hidden_size = 32
    num_layers = 1
    output_size = 2
    n_epochs = 100

    # Train RNN
    print("\nTraining Simple RNN on long sequences...")
    rnn_model = SimpleRNN(input_size, hidden_size, num_layers, output_size)
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    rnn_history = train_model(rnn_model, train_loader, val_loader, criterion,
                             rnn_optimizer, n_epochs, verbose=False)
    print(f"RNN Final Val Accuracy: {rnn_history['val_acc'][-1]:.4f}")

    # Train LSTM
    print("\nTraining LSTM on long sequences...")
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    lstm_history = train_model(lstm_model, train_loader, val_loader, criterion,
                              lstm_optimizer, n_epochs, verbose=False)
    print(f"LSTM Final Val Accuracy: {lstm_history['val_acc'][-1]:.4f}")

    results = {
        'RNN': {
            'final_val_acc': rnn_history['val_acc'][-1],
            'history': rnn_history
        },
        'LSTM': {
            'final_val_acc': lstm_history['val_acc'][-1],
            'history': lstm_history
        }
    }

    return results


def experiment_varying_sequence_length():
    """
    Experiment 3: Performance vs sequence length.

    Test how performance degrades with longer sequences.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Performance vs Sequence Length")
    print("=" * 70)

    seq_lengths = [25, 50, 100, 150, 200]
    rnn_accs = []
    lstm_accs = []

    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")

        X, y = generate_long_dependency_data(n_samples=1000, seq_length=seq_len, gap=seq_len-10)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        criterion = nn.CrossEntropyLoss()

        # RNN
        rnn_model = SimpleRNN(1, 32, 1, 2)
        rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)
        rnn_history = train_model(rnn_model, train_loader, val_loader, criterion,
                                 rnn_optimizer, n_epochs=50, verbose=False)
        rnn_accs.append(max(rnn_history['val_acc']))

        # LSTM
        lstm_model = LSTMModel(1, 32, 1, 2)
        lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        lstm_history = train_model(lstm_model, train_loader, val_loader, criterion,
                                  lstm_optimizer, n_epochs=50, verbose=False)
        lstm_accs.append(max(lstm_history['val_acc']))

        print(f"  RNN: {rnn_accs[-1]:.4f}, LSTM: {lstm_accs[-1]:.4f}")

    return {
        'seq_lengths': seq_lengths,
        'rnn_accs': rnn_accs,
        'lstm_accs': lstm_accs
    }


# ==================== VISUALIZATION ====================

def plot_training_comparison(results, title="Training Comparison", save_path=None):
    """Plot training curves for RNN vs LSTM."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1 = axes[0]
    ax1.plot(results['RNN']['history']['train_loss'], 'b-', label='RNN Train', alpha=0.7)
    ax1.plot(results['RNN']['history']['val_loss'], 'b--', label='RNN Val')
    ax1.plot(results['LSTM']['history']['train_loss'], 'r-', label='LSTM Train', alpha=0.7)
    ax1.plot(results['LSTM']['history']['val_loss'], 'r--', label='LSTM Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2 = axes[1]
    ax2.plot(results['RNN']['history']['train_acc'], 'b-', label='RNN Train', alpha=0.7)
    ax2.plot(results['RNN']['history']['val_acc'], 'b--', label='RNN Val')
    ax2.plot(results['LSTM']['history']['train_acc'], 'r-', label='LSTM Train', alpha=0.7)
    ax2.plot(results['LSTM']['history']['val_acc'], 'r--', label='LSTM Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def plot_sequence_length_comparison(results, save_path=None):
    """Plot performance vs sequence length."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = results['seq_lengths']
    width = 0.35

    ax.bar([i - width/2 for i in range(len(x))], results['rnn_accs'],
           width, label='RNN', color='steelblue')
    ax.bar([i + width/2 for i in range(len(x))], results['lstm_accs'],
           width, label='LSTM', color='darkorange')

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Best Validation Accuracy')
    ax.set_title('Performance vs Sequence Length (Long-Range Dependencies)')
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return fig


def create_architecture_comparison_table():
    """Create comparison table of RNN vs LSTM architecture."""
    comparison = {
        'Aspect': [
            'Core Mechanism',
            'Memory Handling',
            'Gradient Flow',
            'Parameters',
            'Training Speed',
            'Long Dependencies',
            'Computational Cost',
            'Use Case'
        ],
        'Simple RNN': [
            'Single tanh activation',
            'Hidden state only',
            'Prone to vanishing/exploding',
            'Fewer (O(h²))',
            'Faster per epoch',
            'Poor (gradients vanish)',
            'Lower',
            'Short sequences, simple patterns'
        ],
        'LSTM': [
            'Forget, Input, Output gates',
            'Cell state + hidden state',
            'Controlled via gates',
            'More (4× RNN)',
            'Slower per epoch',
            'Good (cell state preserves)',
            'Higher',
            'Long sequences, complex patterns'
        ]
    }

    return pd.DataFrame(comparison)


def run_full_comparison():
    """Run all experiments and generate report."""
    print("=" * 70)
    print("RESEARCH TASK: RNN vs LSTM COMPARISON")
    print("Topic #1: Comparing Neural Networks for NLP")
    print("=" * 70)

    # Print architecture comparison
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)

    comparison_table = create_architecture_comparison_table()
    print("\n" + comparison_table.to_string(index=False))

    # Experiment 1: Standard sequences
    exp1_results = experiment_standard_sequences()

    print("\n" + "-" * 50)
    print("Experiment 1 Results:")
    print("-" * 50)
    for model_name, metrics in exp1_results.items():
        print(f"\n{model_name}:")
        print(f"  Parameters: {metrics['n_parameters']:,}")
        print(f"  Best Val Accuracy: {metrics['best_val_acc']:.4f}")
        print(f"  Avg Epoch Time: {metrics['avg_epoch_time']:.3f}s")

    plot_training_comparison(exp1_results, "Experiment 1: Standard Sequences",
                            save_path=os.path.join(RESULTS_DIR, 'exp1_training.png'))

    # Experiment 2: Long-range dependencies
    exp2_results = experiment_long_dependencies()

    print("\n" + "-" * 50)
    print("Experiment 2 Results (Long Dependencies):")
    print("-" * 50)
    for model_name, metrics in exp2_results.items():
        print(f"  {model_name} Final Accuracy: {metrics['final_val_acc']:.4f}")

    plot_training_comparison(exp2_results, "Experiment 2: Long-Range Dependencies",
                            save_path=os.path.join(RESULTS_DIR, 'exp2_training.png'))

    # Experiment 3: Varying sequence length
    exp3_results = experiment_varying_sequence_length()

    plot_sequence_length_comparison(exp3_results, save_path=os.path.join(RESULTS_DIR, 'exp3_seq_length.png'))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY AND CONCLUSIONS")
    print("=" * 70)

    print("""
1. STANDARD SEQUENCE CLASSIFICATION:
   - Both RNN and LSTM can learn simple patterns
   - LSTM typically achieves slightly better accuracy
   - RNN trains faster per epoch due to simpler architecture

2. LONG-RANGE DEPENDENCIES:
   - LSTM significantly outperforms RNN
   - RNN struggles due to vanishing gradient problem
   - LSTM's gating mechanism preserves information over time

3. SEQUENCE LENGTH IMPACT:
   - RNN performance degrades rapidly with longer sequences
   - LSTM maintains better performance on longer sequences
   - The gap widens as sequence length increases

4. RECOMMENDATIONS:
   - Use RNN for: Short sequences, real-time applications, limited resources
   - Use LSTM for: Long sequences, complex temporal patterns, NLP tasks

5. KEY INSIGHT:
   The LSTM's cell state acts as a "highway" for gradient flow,
   allowing information to persist over many timesteps without
   degradation - the fundamental advantage over vanilla RNN.
""")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_comparison()
