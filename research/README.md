# Research Task: Comparison of RNN and LSTM Neural Networks

## Topic #1
**Comparison of RNN and LSTM Neural Network Types for Natural Language Processing**

## Structure

### Report Requirements
- Format: Text report (10-20 pages)
- Font: 14pt, clear formatting
- Language: English or Ukrainian

### Report Structure
1. Abstract (up to 5 sentences)
2. Introduction
3. Problem Statement
4. Review of Models and Methods (theoretical justification)
5. Experimental Results (for high grade)
6. Analysis of Results
7. Conclusions
8. References (cited in text)

## Files

### Implementation
- `rnn_lstm_comparison.py` - Complete experimental comparison
  - Simple RNN implementation
  - LSTM implementation
  - Training/evaluation pipeline
  - Visualization functions
  - Three experiments:
    1. Standard sequence classification
    2. Long-range dependency test
    3. Sequence length scaling analysis

### Documentation
- `REPORT.md` - Full research report in markdown format
- `README.md` - This file

## Usage

```bash
source venv/bin/activate
cd research
python rnn_lstm_comparison.py
```

## Key Findings

| Aspect | RNN | LSTM |
|--------|-----|------|
| Long-range dependencies | Poor (vanishing gradients) | Good (gating mechanism) |
| Parameters | Fewer (~h²) | More (~4h²) |
| Training speed | Faster | Slower |
| Short sequences | Good | Good |
| Long sequences | Degrades rapidly | Maintains performance |

## Output Files
- `exp1_training.png` - Standard sequence training curves
- `exp2_training.png` - Long-range dependency results
- `exp3_seq_length.png` - Performance vs sequence length

## Dependencies
- torch (PyTorch)
- numpy
- matplotlib
