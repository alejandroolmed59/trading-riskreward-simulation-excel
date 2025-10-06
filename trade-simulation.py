# Creating a 200-trade simulation Excel with three sheets: Fixed Risk, Martingale (soft increase after loss), Anti-Martingale (increase after win).
# Assumptions (explicit in the file as well):
# - Starting capital: $10,000
# - Base risk per trade: 1% of current capital
# - Win probability: 40%
# - Reward-to-risk on winners: 2:1 (i.e., a win returns +2 * risk_amount)
# - Martingale: after a loss increase risk% by +1 percentage point (absolute), reset to base on win. If risk% exceeds 100% it is treated as all-in (capped at 100%).
# - Anti-martingale: after a win increase risk% by +1 percentage point, reset to base on loss. Same all-in cap applies.
# - Outcomes are the same random sequence across strategies for direct comparison (seeded for reproducibility).
# The script saves an Excel workbook to /mnt/data/trading_simulation.xlsx and displays the first rows of each sheet.

import numpy as np
import pandas as pd
from pathlib import Path
np.random.seed(478)

# Parameters
starting_capital = 10000.0
n_trades = 200
base_risk_pct = 0.01  # 1%
win_prob = 0.40
reward_factor = 2.0  # winners pay 2:1 relative to risk amount
step_pct = 0.01  # 1 percentage point increments for martingale/anti-martingale

# Generate outcomes (True = win, False = loss)
outcomes = np.random.rand(n_trades) < win_prob

def run_fixed(capital0, outcomes, base_risk_pct, reward_factor):
    rows = []
    capital = capital0
    for i, win in enumerate(outcomes, start=1):
        risk_pct = base_risk_pct
        risk_amount = capital * risk_pct
        pnl = risk_amount * reward_factor if win else -risk_amount
        capital += pnl
        rows.append({
            "Trade": i,
            "Outcome": "Win" if win else "Loss",
            "Risk%": risk_pct,
            "RiskAmount": round(risk_amount, 2),
            "P&L": round(pnl, 2),
            "Capital": round(capital, 2)
        })
    return pd.DataFrame(rows)

def run_martingale(capital0, outcomes, base_risk_pct, reward_factor, step_pct):
    rows = []
    capital = capital0
    risk_pct_abs = base_risk_pct  # absolute percentage (e.g., 0.01, 0.02, ...)
    for i, win in enumerate(outcomes, start=1):
        # cap at 100% (all-in) to avoid absurd percentages
        if risk_pct_abs > 1.0:
            risk_pct_abs = 1.0
        risk_amount = capital * risk_pct_abs
        pnl = risk_amount * reward_factor if win else -risk_amount
        capital += pnl
        rows.append({
            "Trade": i,
            "Outcome": "Win" if win else "Loss",
            "Risk%": round(risk_pct_abs, 4),
            "RiskAmount": round(risk_amount, 2),
            "P&L": round(pnl, 2),
            "Capital": round(capital, 2)
        })
        # update risk_pct_abs: after loss increase by step, after win reset to base
        if win:
            risk_pct_abs = base_risk_pct
        else:
            risk_pct_abs += step_pct
    return pd.DataFrame(rows)

def run_anti_martingale(capital0, outcomes, base_risk_pct, reward_factor, step_pct):
    rows = []
    capital = capital0
    risk_pct_abs = base_risk_pct
    for i, win in enumerate(outcomes, start=1):
        if risk_pct_abs > 1.0:
            risk_pct_abs = 1.0
        risk_amount = capital * risk_pct_abs
        pnl = risk_amount * reward_factor if win else -risk_amount
        capital += pnl
        rows.append({
            "Trade": i,
            "Outcome": "Win" if win else "Loss",
            "Risk%": round(risk_pct_abs, 4),
            "RiskAmount": round(risk_amount, 2),
            "P&L": round(pnl, 2),
            "Capital": round(capital, 2)
        })
        # update: after win increase, after loss reset to base
        if win:
            risk_pct_abs += step_pct
        else:
            risk_pct_abs = base_risk_pct
    return pd.DataFrame(rows)

# Run simulations
fixed_df = run_fixed(starting_capital, outcomes, base_risk_pct, reward_factor)
martingale_df = run_martingale(starting_capital, outcomes, base_risk_pct, reward_factor, step_pct)
anti_df = run_anti_martingale(starting_capital, outcomes, base_risk_pct, reward_factor, step_pct)

# Save to Excel
out_path = Path("./trading_simulation.xlsx")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    fixed_df.to_excel(writer, sheet_name="Fixed_Risk_1pct", index=False)
    martingale_df.to_excel(writer, sheet_name="Martingale_soft", index=False)
    anti_df.to_excel(writer, sheet_name="Anti_Martingale", index=False)

print("Saved Excel to:", out_path)
