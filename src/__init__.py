"""
Nissan Commodity Risk Analysis Engine

A production-ready Monte Carlo simulation framework for commodity price risk modeling.
"""

__version__ = '1.0.0'
__author__ = 'Quantitative Risk Team'

from .portfolio import Commodity, CommodityPortfolio
from .simulation import MonteCarloEngine
from .analytics import calculate_risk_metrics, fit_evt_gpd

__all__ = [
    'Commodity',
    'CommodityPortfolio', 
    'MonteCarloEngine',
    'calculate_risk_metrics',
    'fit_evt_gpd'
]