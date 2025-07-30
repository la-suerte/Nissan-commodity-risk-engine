"""
Analytics module for risk metrics and extreme value analysis.
Contains functions for VaR, ES, and EVT calculations.
"""
import numpy as np
from scipy.stats import genpareto
from typing import Tuple


def calculate_risk_metrics(cost_overruns: np.ndarray, 
                          confidence_level: float) -> Tuple[float, float]:
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES).
    
    Args:
        cost_overruns: Array of cost overruns (simulated - baseline)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Tuple of (VaR, ES) at the specified confidence level
    """
    # Value at Risk: quantile at confidence level
    var = np.quantile(cost_overruns, confidence_level)
    
    # Expected Shortfall: mean of losses beyond VaR
    es = cost_overruns[cost_overruns > var].mean()
    
    return var, es


def fit_evt_gpd(cost_overruns: np.ndarray, 
                threshold_quantile: float = 0.90) -> Tuple[float, float, float]:
    """
    Perform Extreme Value Theory (EVT) analysis using Generalized Pareto Distribution.
    
    Fits GPD to tail exceedances above a threshold.
    
    Args:
        cost_overruns: Array of cost overruns
        threshold_quantile: Quantile to use as threshold (default 90th percentile)
        
    Returns:
        Tuple of (shape, scale, evt_var_995)
        - shape: GPD shape parameter
        - scale: GPD scale parameter  
        - evt_var_995: EVT-based 99.5% VaR
    """
    # Determine threshold at specified quantile
    threshold = np.quantile(cost_overruns, threshold_quantile)
    
    # Calculate exceedances above threshold
    exceedances = cost_overruns[cost_overruns > threshold] - threshold
    
    # Fit Generalized Pareto Distribution
    # floc=0 fixes location at 0 (exceedances start at 0)
    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    
    # Calculate EVT-based VaR at 99.5% confidence
    # Formula: VaR = u + (scale/shape) * [((1-q)/P(X>u))^(-shape) - 1]
    q = 0.995  # 99.5% confidence
    prob_tail = 1.0 - threshold_quantile  # Probability in tail
    
    if shape != 0:
        evt_var = threshold + (scale / shape) * (
            ((1 - q) / prob_tail) ** (-shape) - 1
        )
    else:
        # Log-normal case (shape = 0)
        evt_var = threshold - scale * np.log((1 - q) / prob_tail)
    
    return shape, scale, evt_var


def calculate_hedging_effectiveness(unhedged_metrics: dict, 
                                   hedged_metrics: dict) -> dict:
    """
    Calculate the effectiveness of hedging strategy.
    
    Args:
        unhedged_metrics: Dict with 'var' and 'es' keys for unhedged portfolio
        hedged_metrics: Dict with 'var' and 'es' keys for hedged portfolio
        
    Returns:
        Dict with risk reduction amounts and percentages
    """
    var_reduction = unhedged_metrics['var'] - hedged_metrics['var']
    es_reduction = unhedged_metrics['es'] - hedged_metrics['es']
    
    var_reduction_pct = (var_reduction / unhedged_metrics['var']) * 100
    es_reduction_pct = (es_reduction / unhedged_metrics['es']) * 100
    
    return {
        'var_reduction': var_reduction,
        'var_reduction_pct': var_reduction_pct,
        'es_reduction': es_reduction,
        'es_reduction_pct': es_reduction_pct
    }