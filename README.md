# Trading Simulation

A Python-based trading simulation that compares three different risk management strategies:
- **Fixed Risk**: Consistent 1% risk per trade
- **Martingale**: Increases risk after losses (soft increase)
- **Anti-Martingale**: Increases risk after wins

## Setup

### 1. Create Virtual Environment

```bash
virtualenv venv
```

### 2. Activate Virtual Environment

```bash
# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:

```bash
python trade-simulation.py
```

This will generate an Excel file with three sheets comparing the different strategies over 200 trades.

## Parameters

- **Starting capital**: $10,000
- **Base risk per trade**: 1% of current capital
- **Win probability**: 40%
- **Reward-to-risk ratio**: 2:1
- **Number of trades**: 200

## Deactivate Virtual Environment

When you're done:

```bash
deactivate
```

