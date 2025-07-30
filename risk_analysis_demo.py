"""
Demo script for Nissan Commodity Risk Analysis Engine.

Demonstrates portfolio initialization, Monte Carlo simulation,
risk metric calculation, and hedging analysis.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.portfolio import CommodityPortfolio
from src.simulation import MonteCarloEngine
from src.analytics import calculate_risk_metrics, fit_evt_gpd, calculate_hedging_effectiveness


def main():
    """Run complete risk analysis demonstration."""
    
    print("=" * 80)
    print("NISSAN COMMODITY RISK ANALYSIS - DEMO")
    print("=" * 80)
    print()
    
    # 1. Initialize Portfolio
    print("1. LOADING PORTFOLIO DATA")
    print("-" * 80)
    portfolio = CommodityPortfolio(data_path='data')
    baseline_cost = portfolio.calculate_baseline_cost()
    
    print(f"Commodities loaded: {len(portfolio.commodities)}")
    print(f"Baseline Annual Cost: ${baseline_cost:,.0f}")
    print()
    
    # 2. Run Monte Carlo Simulation
    print("2. RUNNING MONTE CARLO SIMULATION")
    print("-" * 80)
    engine = MonteCarloEngine(portfolio)
    
    N_SIMS = 10000
    TIME_HORIZON = 1.0
    
    print(f"Simulations: {N_SIMS:,}")
    print(f"Time Horizon: {TIME_HORIZON} year")
    
    simulated_costs = engine.run_simulation(n_sims=N_SIMS, time_horizon=TIME_HORIZON)
    cost_overruns = simulated_costs - baseline_cost
    
    print(f"Mean Simulated Cost: ${simulated_costs.mean():,.0f}")
    print(f"Std Dev: ${simulated_costs.std():,.0f}")
    print()
    
    # 3. Calculate Risk Metrics
    print("3. RISK METRICS (UNHEDGED)")
    print("-" * 80)
    
    var_95, es_95 = calculate_risk_metrics(cost_overruns, 0.95)
    var_99, es_99 = calculate_risk_metrics(cost_overruns, 0.99)
    
    print(f"95% VaR: ${var_95:,.0f}")
    print(f"95% ES:  ${es_95:,.0f}")
    print()
    print(f"99% VaR: ${var_99:,.0f}")
    print(f"99% ES:  ${es_99:,.0f}")
    print()
    
    # 4. Extreme Value Theory
    print("4. EXTREME VALUE THEORY (EVT)")
    print("-" * 80)
    
    shape, scale, evt_var_995 = fit_evt_gpd(cost_overruns, threshold_quantile=0.90)
    
    print(f"GPD Shape Parameter: {shape:.4f}")
    print(f"GPD Scale Parameter: ${scale:,.0f}")
    print(f"EVT 99.5% VaR: ${evt_var_995:,.0f}")
    print()
    
    # 5. Hedging Analysis
    print("5. HEDGING STRATEGY ANALYSIS")
    print("-" * 80)
    
    HEDGE_RATIO = 0.60
    HEDGE_COMMODITIES = ['Steel', 'Aluminum']
    
    print(f"Hedge Ratio: {HEDGE_RATIO:.0%}")
    print(f"Hedged Commodities: {', '.join(HEDGE_COMMODITIES)}")
    print()
    
    hedged_costs = engine.run_hedged_simulation(
        n_sims=N_SIMS,
        time_horizon=TIME_HORIZON,
        hedge_ratio=HEDGE_RATIO,
        hedge_commodities=HEDGE_COMMODITIES
    )
    
    hedged_overruns = hedged_costs - baseline_cost
    var_95_hedged, es_95_hedged = calculate_risk_metrics(hedged_overruns, 0.95)
    
    # Calculate effectiveness
    unhedged = {'var': var_95, 'es': es_95}
    hedged = {'var': var_95_hedged, 'es': es_95_hedged}
    effectiveness = calculate_hedging_effectiveness(unhedged, hedged)
    
    print("RESULTS:")
    print(f"  Unhedged 95% VaR: ${var_95:,.0f}")
    print(f"  Hedged 95% VaR:   ${var_95_hedged:,.0f}")
    print(f"  Risk Reduction:   ${effectiveness['var_reduction']:,.0f} ({effectiveness['var_reduction_pct']:.1f}%)")
    print()
    print(f"  Unhedged 95% ES:  ${es_95:,.0f}")
    print(f"  Hedged 95% ES:    ${es_95_hedged:,.0f}")
    print(f"  Risk Reduction:   ${effectiveness['es_reduction']:,.0f} ({effectiveness['es_reduction_pct']:.1f}%)")
    print()
    
    # 6. Summary
    print("=" * 80)
    print("EXECUTIVE SUMMARY")
    print("=" * 80)
    print(f"Total Annual Exposure: ${baseline_cost:,.0f}")
    print(f"95% VaR (Unhedged): ${var_95:,.0f}")
    print(f"99.5% EVT VaR: ${evt_var_995:,.0f}")
    print()
    print(f"RECOMMENDATION: {HEDGE_RATIO:.0%} hedging on {', '.join(HEDGE_COMMODITIES)}")
    print(f"reduces 95% ES tail risk by ${effectiveness['es_reduction']:,.0f}")
    print("=" * 80)


if __name__ == '__main__':
    main()