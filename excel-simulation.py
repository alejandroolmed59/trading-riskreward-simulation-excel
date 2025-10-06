# Creating a 100-trade simulation Excel with four sheets: Fixed Risk, Martingale (soft increase after loss), Anti-Martingale (increase after win), and Summary.
# Assumptions (explicit in the file as well):
# - Starting capital: $10,000
# - Base risk per trade: 1% of current capital
# - Win probability: 45%
# - Reward-to-risk on winners: 1.5:1 (i.e., a win returns +1.5 * risk_amount)
# - Martingale: after a loss increase risk% by +1 percentage point (absolute), reset to base on win. If risk% exceeds 100% it is treated as all-in (capped at 100%).
# - Anti-martingale: after a win increase risk% by +1 percentage point, reset to base on loss. Same all-in cap applies.
# - Outcomes are the same random sequence across strategies for direct comparison (seeded for reproducibility).
# The script saves an Excel workbook with a Summary sheet showing key statistics including max drawdown.

import numpy as np
import pandas as pd
from pathlib import Path

def run_fixed(capital0, outcomes, base_risk_pct, reward_factor):
    """Run fixed risk strategy simulation"""
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
    """Run martingale strategy simulation"""
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
    """Run anti-martingale strategy simulation"""
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

def calculate_stats(df, starting_capital, strategy_name):
    """Calculate comprehensive statistics for a trading strategy"""
    capital_series = df["Capital"].values
    pnl_series = df["P&L"].values
    
    # Calculate running maximum (peak) for drawdown calculation
    running_max = np.maximum.accumulate(capital_series)
    drawdown = running_max - capital_series
    drawdown_pct = (drawdown / running_max) * 100
    
    # Basic stats
    final_capital = capital_series[-1]
    total_return = final_capital - starting_capital
    total_return_pct = (total_return / starting_capital) * 100
    
    # Win/Loss stats
    wins = df[df["Outcome"] == "Win"]
    losses = df[df["Outcome"] == "Loss"]
    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = (num_wins / len(df)) * 100 if len(df) > 0 else 0
    
    # P&L stats
    avg_win = wins["P&L"].mean() if num_wins > 0 else 0
    avg_loss = losses["P&L"].mean() if num_losses > 0 else 0
    best_trade = pnl_series.max()
    worst_trade = pnl_series.min()
    
    # Drawdown stats
    max_drawdown = drawdown.max()
    max_drawdown_pct = drawdown_pct.max()
    
    return {
        "Strategy": strategy_name,
        "Starting Capital": f"${starting_capital:,.2f}",
        "Final Capital": f"${final_capital:,.2f}",
        "Total Return": f"${total_return:,.2f}",
        "Total Return %": f"{total_return_pct:.2f}%",
        "Max Drawdown": f"${max_drawdown:,.2f}",
        "Max Drawdown %": f"{max_drawdown_pct:.2f}%",
        "Total Trades": len(df),
        "Wins": num_wins,
        "Losses": num_losses,
        "Win Rate %": f"{win_rate:.2f}%",
        "Avg Win": f"${avg_win:.2f}",
        "Avg Loss": f"${avg_loss:.2f}",
        "Best Trade": f"${best_trade:.2f}",
        "Worst Trade": f"${worst_trade:.2f}"
    }

def run_excel_simulation(seed=552, starting_capital=10000.0, n_trades=100, 
                        base_risk_pct=0.01, win_prob=0.45, reward_factor=1.5, step_pct=0.01,
                        save_excel=True, output_dir="./"):
    """
    Run a single trading simulation and optionally save to Excel.
    
    Args:
        seed: Random seed for reproducibility
        starting_capital: Starting capital amount
        n_trades: Number of trades to simulate
        base_risk_pct: Base risk percentage per trade
        win_prob: Probability of winning a trade
        reward_factor: Reward factor for winning trades
        step_pct: Step percentage for martingale strategies
        save_excel: Whether to save results to Excel file
        output_dir: Directory to save Excel file
    
    Returns:
        Dictionary containing all simulation results and DataFrames
    """
    print(f"=== EXCEL TRADING SIMULATION ===")
    print(f"Seed: {seed}")
    print(f"Starting Capital: ${starting_capital:,.2f}")
    print(f"Number of Trades: {n_trades}")
    print(f"Win Probability: {win_prob:.1%}")
    print(f"Base Risk: {base_risk_pct:.1%}")
    print(f"Reward Factor: {reward_factor}:1")
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Generate outcomes (True = win, False = loss)
    outcomes = np.random.rand(n_trades) < win_prob
    
    # Run simulations
    print(f"\nRunning simulations...")
    fixed_df = run_fixed(starting_capital, outcomes, base_risk_pct, reward_factor)
    martingale_df = run_martingale(starting_capital, outcomes, base_risk_pct, reward_factor, step_pct)
    anti_df = run_anti_martingale(starting_capital, outcomes, base_risk_pct, reward_factor, step_pct)
    
    # Calculate statistics for all strategies
    stats_list = [
        calculate_stats(fixed_df, starting_capital, "Fixed Risk (1%)"),
        calculate_stats(martingale_df, starting_capital, "Martingale (Soft)"),
        calculate_stats(anti_df, starting_capital, "Anti-Martingale"),
    ]
    summary_df = pd.DataFrame(stats_list)
    
    # Print summary
    print(f"\n=== SIMULATION SUMMARY ===")
    print(summary_df.to_string(index=False))
    
    # Save to Excel if requested
    if save_excel:
        output_path = Path(output_dir) / f"trading_simulation_{seed}.xlsx"
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            fixed_df.to_excel(writer, sheet_name="Fixed_Risk_1pct", index=False)
            martingale_df.to_excel(writer, sheet_name="Martingale_soft", index=False)
            anti_df.to_excel(writer, sheet_name="Anti_Martingale", index=False)
        
        print(f"\nSaved Excel to: {output_path}")
    
    return {
        'summary_df': summary_df,
        'fixed_df': fixed_df,
        'martingale_df': martingale_df,
        'anti_df': anti_df,
        'outcomes': outcomes,
        'parameters': {
            'seed': seed,
            'starting_capital': starting_capital,
            'n_trades': n_trades,
            'base_risk_pct': base_risk_pct,
            'win_prob': win_prob,
            'reward_factor': reward_factor,
            'step_pct': step_pct
        }
    }

if __name__ == "__main__":
    # Run the Excel simulation with default parameters
    results = run_excel_simulation(
        seed=552,
        starting_capital=10000.0,
        n_trades=100,
        base_risk_pct=0.01,
        win_prob=0.45,
        reward_factor=1.5,
        step_pct=0.01,
        save_excel=True
    )
