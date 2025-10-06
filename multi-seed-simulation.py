# Multi-Seed Trading Simulation
# This script runs multiple trading simulations with different random seeds to determine
# the best performing strategy across various market conditions.
#
# Features:
# - Tests multiple seeds (default: 50) with 100 trades each
# - Compares Fixed Risk, Martingale, and Anti-Martingale strategies
# - Provides comprehensive statistical analysis and rankings
# - Identifies the best performing strategy based on multiple criteria

import numpy as np
import pandas as pd

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Custom colors
    PURPLE = '\033[35m'
    BLUE = '\033[34m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

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

def run_multi_seed_simulation(num_seeds=50, seedOffset=0, n_trades=100, starting_capital=10000.0, 
                            base_risk_pct=0.01, win_prob=0.45, reward_factor=1.5, step_pct=0.01):
    """
    Run multiple trading simulations with different seeds to determine the best performing strategy.
    
    Args:
        num_seeds: Number of different seeds to test
        seedOffset: Offset for seed values (allows testing different ranges)
        n_trades: Number of trades per simulation
        starting_capital: Starting capital for each simulation
        base_risk_pct: Base risk percentage
        win_prob: Win probability
        reward_factor: Reward factor for winning trades
        step_pct: Step percentage for martingale strategies
    
    Returns:
        Dictionary containing aggregated results and best strategy analysis
    """
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== RUNNING MULTI-SEED SIMULATION ==={Colors.ENDC}")
    print(f"{Colors.OKCYAN}Testing {num_seeds} different seeds with {n_trades} trades each{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Seed range:{Colors.ENDC} {Colors.WHITE}{seedOffset} to {seedOffset + num_seeds - 1}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Starting capital:{Colors.ENDC} {Colors.GREEN}${starting_capital:,.2f}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Win probability:{Colors.ENDC} {Colors.YELLOW}{win_prob:.1%}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Base risk:{Colors.ENDC} {Colors.RED}{base_risk_pct:.1%}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Reward factor:{Colors.ENDC} {Colors.BLUE}{reward_factor}:1{Colors.ENDC}")
    
    # Store results for each strategy across all seeds
    all_results = {
        'Fixed Risk': [],
        'Martingale': [],
        'Anti-Martingale': []
    }
    
    # Run simulations for each seed
    for seed_idx in range(num_seeds):
        np.random.seed(seed_idx + seedOffset)
        
        # Generate outcomes for this seed
        outcomes = np.random.rand(n_trades) < win_prob
        
        # Run all three strategies
        fixed_df = run_fixed(starting_capital, outcomes, base_risk_pct, reward_factor)
        martingale_df = run_martingale(starting_capital, outcomes, base_risk_pct, reward_factor, step_pct)
        anti_df = run_anti_martingale(starting_capital, outcomes, base_risk_pct, reward_factor, step_pct)
        
        # Calculate stats for each strategy
        fixed_stats = calculate_stats(fixed_df, starting_capital, "Fixed Risk")
        martingale_stats = calculate_stats(martingale_df, starting_capital, "Martingale")
        anti_stats = calculate_stats(anti_df, starting_capital, "Anti-Martingale")
        
        # Store final capital and return percentage for comparison
        all_results['Fixed Risk'].append({
            'seed': seed_idx + seedOffset,
            'final_capital': fixed_df['Capital'].iloc[-1],
            'total_return_pct': float(fixed_stats['Total Return %'].replace('%', '')),
            'max_drawdown_pct': float(fixed_stats['Max Drawdown %'].replace('%', '')),
            'win_rate': float(fixed_stats['Win Rate %'].replace('%', ''))
        })
        
        all_results['Martingale'].append({
            'seed': seed_idx + seedOffset,
            'final_capital': martingale_df['Capital'].iloc[-1],
            'total_return_pct': float(martingale_stats['Total Return %'].replace('%', '')),
            'max_drawdown_pct': float(martingale_stats['Max Drawdown %'].replace('%', '')),
            'win_rate': float(martingale_stats['Win Rate %'].replace('%', ''))
        })
        
        all_results['Anti-Martingale'].append({
            'seed': seed_idx + seedOffset,
            'final_capital': anti_df['Capital'].iloc[-1],
            'total_return_pct': float(anti_stats['Total Return %'].replace('%', '')),
            'max_drawdown_pct': float(anti_stats['Max Drawdown %'].replace('%', '')),
            'win_rate': float(anti_stats['Win Rate %'].replace('%', ''))
        })
        
        # Progress indicator
        if (seed_idx + 1) % 10 == 0:
            progress_pct = ((seed_idx + 1) / num_seeds) * 100
            print(f"{Colors.OKBLUE}Completed {seed_idx + 1}/{num_seeds} simulations... ({progress_pct:.0f}%){Colors.ENDC}")
    
    return analyze_strategy_performance(all_results)

def analyze_strategy_performance(all_results):
    """
    Analyze and compare strategy performance across all seeds.
    
    Args:
        all_results: Dictionary containing results for each strategy
    
    Returns:
        Dictionary with performance analysis and rankings
    """
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== STRATEGY PERFORMANCE ANALYSIS ==={Colors.ENDC}")
    
    strategy_summary = {}
    
    for strategy_name, results in all_results.items():
        # Extract metrics
        returns = [r['total_return_pct'] for r in results]
        final_capitals = [r['final_capital'] for r in results]
        max_drawdowns = [r['max_drawdown_pct'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        
        # Calculate statistics
        avg_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)
        min_return = np.min(returns)
        max_return = np.max(returns)
        
        avg_final_capital = np.mean(final_capitals)
        median_final_capital = np.median(final_capitals)
        
        avg_drawdown = np.mean(max_drawdowns)
        max_drawdown = np.max(max_drawdowns)
        
        avg_win_rate = np.mean(win_rates)
        
        # Count positive returns
        positive_returns = sum(1 for r in returns if r > 0)
        positive_rate = (positive_returns / len(returns)) * 100
        
        # Count returns above 10%
        high_returns = sum(1 for r in returns if r > 10)
        high_return_rate = (high_returns / len(returns)) * 100
        
        strategy_summary[strategy_name] = {
            'avg_return': avg_return,
            'median_return': median_return,
            'std_return': std_return,
            'min_return': min_return,
            'max_return': max_return,
            'avg_final_capital': avg_final_capital,
            'median_final_capital': median_final_capital,
            'avg_drawdown': avg_drawdown,
            'max_drawdown': max_drawdown,
            'avg_win_rate': avg_win_rate,
            'positive_rate': positive_rate,
            'high_return_rate': high_return_rate,
            'num_simulations': len(results)
        }
        
        # Print summary for this strategy
        print(f"\n{Colors.OKCYAN}{Colors.BOLD}--- {strategy_name} ---{Colors.ENDC}")
        
        # Color code returns based on performance
        avg_return_color = Colors.GREEN if avg_return > 0 else Colors.RED
        median_return_color = Colors.GREEN if median_return > 0 else Colors.RED
        positive_rate_color = Colors.GREEN if positive_rate > 50 else Colors.YELLOW if positive_rate > 30 else Colors.RED
        
        print(f"{Colors.WHITE}Average Return:{Colors.ENDC} {avg_return_color}{avg_return:.2f}%{Colors.ENDC}")
        print(f"{Colors.WHITE}Median Return:{Colors.ENDC} {median_return_color}{median_return:.2f}%{Colors.ENDC}")
        print(f"{Colors.WHITE}Return Std Dev:{Colors.ENDC} {Colors.YELLOW}{std_return:.2f}%{Colors.ENDC}")
        print(f"{Colors.WHITE}Return Range:{Colors.ENDC} {Colors.RED}{min_return:.2f}%{Colors.ENDC} to {Colors.GREEN}{max_return:.2f}%{Colors.ENDC}")
        print(f"{Colors.WHITE}Average Final Capital:{Colors.ENDC} {Colors.GREEN}${avg_final_capital:,.2f}{Colors.ENDC}")
        print(f"{Colors.WHITE}Average Max Drawdown:{Colors.ENDC} {Colors.RED}{avg_drawdown:.2f}%{Colors.ENDC}")
        print(f"{Colors.WHITE}Worst Max Drawdown:{Colors.ENDC} {Colors.RED}{max_drawdown:.2f}%{Colors.ENDC}")
        print(f"{Colors.WHITE}Positive Returns:{Colors.ENDC} {positive_rate_color}{positive_rate:.1f}%{Colors.ENDC} ({positive_returns}/{len(results)})")
        print(f"{Colors.WHITE}High Returns (>10%):{Colors.ENDC} {Colors.GREEN}{high_return_rate:.1f}%{Colors.ENDC} ({high_returns}/{len(results)})")
    
    # Determine best strategy based on multiple criteria
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== STRATEGY RANKINGS ==={Colors.ENDC}")
    
    # Create ranking based on average return
    avg_return_ranking = sorted(strategy_summary.items(), key=lambda x: x[1]['avg_return'], reverse=True)
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}Ranking by Average Return:{Colors.ENDC}")
    for i, (strategy, stats) in enumerate(avg_return_ranking, 1):
        return_color = Colors.GREEN if stats['avg_return'] > 0 else Colors.RED
        medal_color = Colors.YELLOW if i == 1 else Colors.WHITE if i == 2 else Colors.RED if i == 3 else Colors.WHITE
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{medal_color}{medal} {strategy}: {return_color}{stats['avg_return']:.2f}%{Colors.ENDC}")
    
    # Create ranking based on median return (more robust to outliers)
    median_return_ranking = sorted(strategy_summary.items(), key=lambda x: x[1]['median_return'], reverse=True)
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}Ranking by Median Return:{Colors.ENDC}")
    for i, (strategy, stats) in enumerate(median_return_ranking, 1):
        return_color = Colors.GREEN if stats['median_return'] > 0 else Colors.RED
        medal_color = Colors.YELLOW if i == 1 else Colors.WHITE if i == 2 else Colors.RED if i == 3 else Colors.WHITE
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{medal_color}{medal} {strategy}: {return_color}{stats['median_return']:.2f}%{Colors.ENDC}")
    
    # Create ranking based on positive return rate
    positive_rate_ranking = sorted(strategy_summary.items(), key=lambda x: x[1]['positive_rate'], reverse=True)
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}Ranking by Positive Return Rate:{Colors.ENDC}")
    for i, (strategy, stats) in enumerate(positive_rate_ranking, 1):
        rate_color = Colors.GREEN if stats['positive_rate'] > 50 else Colors.YELLOW if stats['positive_rate'] > 30 else Colors.RED
        medal_color = Colors.YELLOW if i == 1 else Colors.WHITE if i == 2 else Colors.RED if i == 3 else Colors.WHITE
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{medal_color}{medal} {strategy}: {rate_color}{stats['positive_rate']:.1f}%{Colors.ENDC}")
    
    # Create ranking based on risk-adjusted return (return / drawdown)
    risk_adjusted_ranking = []
    for strategy, stats in strategy_summary.items():
        if stats['avg_drawdown'] > 0:
            risk_adjusted_return = stats['avg_return'] / stats['avg_drawdown']
        else:
            risk_adjusted_return = float('inf') if stats['avg_return'] > 0 else 0
        risk_adjusted_ranking.append((strategy, risk_adjusted_return))
    
    risk_adjusted_ranking.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}Ranking by Risk-Adjusted Return (Avg Return / Avg Drawdown):{Colors.ENDC}")
    for i, (strategy, ratio) in enumerate(risk_adjusted_ranking, 1):
        medal_color = Colors.YELLOW if i == 1 else Colors.WHITE if i == 2 else Colors.RED if i == 3 else Colors.WHITE
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        if ratio == float('inf'):
            print(f"{medal_color}{medal} {strategy}: {Colors.GREEN}∞ (no drawdown){Colors.ENDC}")
        else:
            ratio_color = Colors.GREEN if ratio > 1 else Colors.YELLOW if ratio > 0 else Colors.RED
            print(f"{medal_color}{medal} {strategy}: {ratio_color}{ratio:.2f}{Colors.ENDC}")
    
    # Overall best strategy recommendation
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== OVERALL RECOMMENDATION ==={Colors.ENDC}")
    best_strategy = avg_return_ranking[0][0]
    best_stats = avg_return_ranking[0][1]
    
    print(f"{Colors.YELLOW}{Colors.BOLD}🏆 BEST PERFORMING STRATEGY: {best_strategy}{Colors.ENDC}")
    
    # Color code the stats
    return_color = Colors.GREEN if best_stats['avg_return'] > 0 else Colors.RED
    positive_rate_color = Colors.GREEN if best_stats['positive_rate'] > 50 else Colors.YELLOW if best_stats['positive_rate'] > 30 else Colors.RED
    
    print(f"{Colors.WHITE}   • Average Return:{Colors.ENDC} {return_color}{best_stats['avg_return']:.2f}%{Colors.ENDC}")
    print(f"{Colors.WHITE}   • Positive Return Rate:{Colors.ENDC} {positive_rate_color}{best_stats['positive_rate']:.1f}%{Colors.ENDC}")
    print(f"{Colors.WHITE}   • Average Final Capital:{Colors.ENDC} {Colors.GREEN}${best_stats['avg_final_capital']:,.2f}{Colors.ENDC}")
    print(f"{Colors.WHITE}   • Average Max Drawdown:{Colors.ENDC} {Colors.RED}{best_stats['avg_drawdown']:.2f}%{Colors.ENDC}")
    
    return {
        'strategy_summary': strategy_summary,
        'rankings': {
            'avg_return': avg_return_ranking,
            'median_return': median_return_ranking,
            'positive_rate': positive_rate_ranking,
            'risk_adjusted': risk_adjusted_ranking
        },
        'best_strategy': best_strategy,
        'best_stats': best_stats
    }

def run_custom_simulation(num_seeds=50, seedOffset=0, n_trades=100, starting_capital=10000.0, 
                         base_risk_pct=0.01, win_prob=0.45, reward_factor=1.5, step_pct=0.01):
    """
    Convenience function to run multi-seed simulation with custom parameters.
    
    Args:
        num_seeds: Number of different seeds to test
        seedOffset: Offset for seed values (allows testing different ranges)
        n_trades: Number of trades per simulation
        starting_capital: Starting capital for each simulation
        base_risk_pct: Base risk percentage
        win_prob: Win probability
        reward_factor: Reward factor for winning trades
        step_pct: Step percentage for martingale strategies
    
    Returns:
        Dictionary containing aggregated results and best strategy analysis
    """
    return run_multi_seed_simulation(
        num_seeds=num_seeds,
        seedOffset=seedOffset,
        n_trades=n_trades,
        starting_capital=starting_capital,
        base_risk_pct=base_risk_pct,
        win_prob=win_prob,
        reward_factor=reward_factor,
        step_pct=step_pct
    )

if __name__ == "__main__":
    # Run the multi-seed simulation with default parameters
    results = run_multi_seed_simulation(
        num_seeds=50, 
        seedOffset=100, 
        n_trades=100,
        starting_capital=10000.0,
        base_risk_pct=0.01,
        win_prob=0.45,
        reward_factor=1.5,
        step_pct=0.01
    )
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}=== SIMULATION COMPLETED ==={Colors.ENDC}")
    print(f"{Colors.WHITE}Best Strategy:{Colors.ENDC} {Colors.YELLOW}{Colors.BOLD}{results['best_strategy']}{Colors.ENDC}")
    
    return_color = Colors.GREEN if results['best_stats']['avg_return'] > 0 else Colors.RED
    positive_rate_color = Colors.GREEN if results['best_stats']['positive_rate'] > 50 else Colors.YELLOW if results['best_stats']['positive_rate'] > 30 else Colors.RED
    
    print(f"{Colors.WHITE}Average Return:{Colors.ENDC} {return_color}{results['best_stats']['avg_return']:.2f}%{Colors.ENDC}")
    print(f"{Colors.WHITE}Positive Return Rate:{Colors.ENDC} {positive_rate_color}{results['best_stats']['positive_rate']:.1f}%{Colors.ENDC}")
