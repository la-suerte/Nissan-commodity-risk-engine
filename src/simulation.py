"""
Monte Carlo simulation engine for commodity price risk.
Implements correlated Geometric Brownian Motion using Cholesky decomposition.
"""
import numpy as np
import pandas as pd
from scipy.linalg import cholesky


class MonteCarloEngine:
    """Monte Carlo simulation engine for correlated commodity prices."""
    
    def __init__(self, portfolio):
        """
        Initialize the simulation engine with a portfolio.
        
        Args:
            portfolio: CommodityPortfolio instance
        """
        self.portfolio = portfolio
        self.n_commodities = len(portfolio.commodities)
    
    def run_simulation(self, n_sims: int, time_horizon: float) -> pd.Series:
        """
        Run Monte Carlo simulation using correlated GBM.
        
        CRITICAL: Uses NumPy vectorization for maximum efficiency.
        
        Args:
            n_sims: Number of simulation paths
            time_horizon: Time horizon in years
            
        Returns:
            Series of simulated total portfolio costs
        """
        # Get portfolio parameters (vectorized)
        vols = self.portfolio.get_volatilities()
        prices = self.portfolio.get_prices()
        volumes = self.portfolio.get_volumes()
        corr_matrix = self.portfolio.correlation_matrix
        
        # Cholesky decomposition for correlation
        L = cholesky(corr_matrix, lower=True)
        
        # Generate uncorrelated random normals (vectorized)
        # Shape: (n_commodities, n_sims)
        Z_uncorr = np.random.normal(0, 1, (self.n_commodities, n_sims))
        
        # Apply correlation structure (vectorized matrix multiplication)
        Z_corr = L @ Z_uncorr
        
        # Geometric Brownian Motion formula (fully vectorized)
        # S_T = S_0 * exp((drift - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        drift = 0.0  # Assume zero drift
        
        # Calculate both terms in a vectorized manner
        # Shape: vols is (n_commodities,), needs to broadcast with (n_commodities, n_sims)
        term1 = (drift - 0.5 * vols**2) * time_horizon  # Shape: (n_commodities,)
        term2 = vols[:, np.newaxis] * np.sqrt(time_horizon) * Z_corr  # Shape: (n_commodities, n_sims)
        
        # Broadcast term1 to match term2's shape
        exponent = term1[:, np.newaxis] + term2
        
        # Calculate simulated end prices (vectorized)
        # Shape: (n_commodities, n_sims)
        sim_prices = prices[:, np.newaxis] * np.exp(exponent)
        
        # Calculate total portfolio costs (vectorized matrix multiplication)
        # volumes @ sim_prices gives (n_sims,) array
        sim_costs = volumes @ sim_prices
        
        return pd.Series(sim_costs)
    
    def run_hedged_simulation(self, n_sims: int, time_horizon: float, 
                            hedge_ratio: float, hedge_commodities: list) -> pd.Series:
        """
        Run simulation with hedging on specific commodities.
        
        Args:
            n_sims: Number of simulation paths
            time_horizon: Time horizon in years
            hedge_ratio: Proportion to hedge (0 to 1)
            hedge_commodities: List of commodity names to hedge
            
        Returns:
            Series of simulated hedged portfolio costs
        """
        # Run base simulation to get simulated prices
        vols = self.portfolio.get_volatilities()
        prices = self.portfolio.get_prices()
        volumes = self.portfolio.get_volumes()
        corr_matrix = self.portfolio.correlation_matrix
        
        # Generate correlated price paths
        L = cholesky(corr_matrix, lower=True)
        Z_uncorr = np.random.normal(0, 1, (self.n_commodities, n_sims))
        Z_corr = L @ Z_uncorr
        
        drift = 0.0
        term1 = (drift - 0.5 * vols**2) * time_horizon
        term2 = vols[:, np.newaxis] * np.sqrt(time_horizon) * Z_corr
        exponent = term1[:, np.newaxis] + term2
        sim_prices = prices[:, np.newaxis] * np.exp(exponent)
        
        # Create hedged cost calculation
        hedged_costs = np.zeros(n_sims)
        commodity_names = self.portfolio.get_commodity_names()
        
        for i, name in enumerate(commodity_names):
            if name in hedge_commodities:
                # Hedged portion uses baseline price, unhedged uses simulated
                hedged_portion = volumes[i] * hedge_ratio * prices[i]
                unhedged_portion = volumes[i] * (1 - hedge_ratio) * sim_prices[i, :]
                hedged_costs += hedged_portion + unhedged_portion
            else:
                # Fully exposed to price movements
                hedged_costs += volumes[i] * sim_prices[i, :]
        
        return pd.Series(hedged_costs)