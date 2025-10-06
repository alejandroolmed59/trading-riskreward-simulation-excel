# Trading Simulation Scripts

This repository contains three Python scripts for trading strategy simulation and analysis.

## Scripts Overview


### 1. `excel-simulation.py`
**Enhanced Excel simulation with configurable parameters**
- Configurable seed, capital, trades, risk parameters
- Optional Excel output
- Function-based design for reusability
- **Usage**: `python excel-simulation.py`

### 2. `multi-seed-simulation.py`
**Multi-seed analysis to determine best performing strategy**
- Tests multiple seeds (default: 50) with 100 trades each
- Comprehensive statistical analysis and rankings
- Identifies best performing strategy across various market conditions
- **Usage**: `python multi-seed-simulation.py`

## Quick Start

### Run Excel Simulation
```bash

# Enhanced simulation
python excel-simulation.py
```

### Run Multi-Seed Analysis
```bash
python multi-seed-simulation.py
```

## Trading Strategies Tested

1. **Fixed Risk (1%)** - Constant 1% risk per trade
2. **Martingale (Soft)** - Increases risk after losses, resets after wins
3. **Anti-Martingale** - Increases risk after wins, resets after losses

## Parameters

- **Starting Capital**: $10,000
- **Win Probability**: 45%
- **Reward Factor**: 1.5:1 (winners pay 1.5x risk amount)
- **Base Risk**: 1% per trade
- **Step Size**: 1% increments for martingale strategies

## Output

### Excel Files
- Detailed trade-by-trade results for each strategy
- Summary statistics including max drawdown
- Multiple sheets for easy analysis

### Multi-Seed Analysis
- Comprehensive statistics for each strategy
- Multiple ranking criteria (average return, median return, positive rate, risk-adjusted)
- Clear identification of best performing strategy
- Progress indicators during execution

## Requirements

- Python 3.7+
- numpy
- pandas
- openpyxl (for Excel output)

## Installation

```bash
pip install numpy pandas openpyxl
```