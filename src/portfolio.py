"""
Portfolio module for commodity exposure modeling.
Contains Commodity and CommodityPortfolio classes.
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path


@dataclass
class Commodity:
    """Dataclass representing a single commodity exposure."""
    name: str
    volume_tons: float
    price_usd_per_ton: float
    annual_volatility: float
    
    @property
    def baseline_cost(self) -> float:
        """Calculate the baseline cost for this commodity."""
        return self.volume_tons * self.price_usd_per_ton


class CommodityPortfolio:
    """Portfolio of commodity exposures with correlation structure."""
    
    def __init__(self, data_path: str = 'data'):
        """
        Initialize portfolio by loading exposure and correlation data.
        
        Args:
            data_path: Path to directory containing CSV files
        """
        self.data_path = Path(data_path)
        self.commodities = []
        self.correlation_matrix = None
        self._load_data()
    
    def _load_data(self):
        """Load commodity exposure and correlation data from CSV files."""
        # Load exposure data
        exposure_df = pd.read_csv(self.data_path / 'nissan_exposure.csv')
        
        for _, row in exposure_df.iterrows():
            commodity = Commodity(
                name=row['Commodity'],
                volume_tons=row['Volume_Tons'],
                price_usd_per_ton=row['Price_USD_per_Ton'],
                annual_volatility=row['Annual_Volatility']
            )
            self.commodities.append(commodity)
        
        # Load correlation matrix
        corr_df = pd.read_csv(self.data_path / 'correlation.csv', index_col=0)
        self.correlation_matrix = corr_df.values
    
    def calculate_baseline_cost(self) -> float:
        """
        Calculate total baseline cost across all commodities.
        
        Returns:
            Total baseline portfolio cost
        """
        return sum(c.baseline_cost for c in self.commodities)
    
    def get_commodity_names(self) -> list:
        """Return list of commodity names."""
        return [c.name for c in self.commodities]
    
    def get_volumes(self) -> np.ndarray:
        """Return array of commodity volumes."""
        return np.array([c.volume_tons for c in self.commodities])
    
    def get_prices(self) -> np.ndarray:
        """Return array of commodity prices."""
        return np.array([c.price_usd_per_ton for c in self.commodities])
    
    def get_volatilities(self) -> np.ndarray:
        """Return array of commodity volatilities."""
        return np.array([c.annual_volatility for c in self.commodities])