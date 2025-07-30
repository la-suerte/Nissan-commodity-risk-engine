"""
Pytest unit tests for the quant risk engine.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.portfolio import CommodityPortfolio
from src.simulation import MonteCarloEngine
from src.analytics import calculate_risk_metrics, fit_evt_gpd


@pytest.fixture
def portfolio():
    """Fixture to create a test portfolio."""
    return CommodityPortfolio(data_path='data')


@pytest.fixture
def engine(portfolio):
    """Fixture to create a test engine."""
    return MonteCarloEngine(portfolio)


class TestPortfolio:
    """Tests for Portfolio module."""
    
    def test_portfolio_loads_correctly(self, portfolio):
        """Test that portfolio data loads without errors."""
        assert len(portfolio.commodities) == 7
        assert portfolio.correlation_matrix is not None
        assert portfolio.correlation_matrix.shape == (7, 7)
    
    def test_baseline_cost_positive(self, portfolio):
        """Test that baseline cost is calculated and positive."""
        baseline = portfolio.calculate_baseline_cost()
        assert baseline > 0
        assert isinstance(baseline, (int, float))
    
    def test_commodity_names(self, portfolio):
        """Test commodity names are retrieved correctly."""
        names = portfolio.get_commodity_names()
        assert len(names) == 7
        assert 'Steel' in names
        assert 'Aluminum' in names


class TestSimulation:
    """Tests for Simulation module."""
    
    def test_simulation_output_dimensions(self, engine):
        """Test that simulation output has correct dimensions."""
        n_sims = 1000
        results = engine.run_simulation(n_sims=n_sims, time_horizon=1.0)
        
        assert len(results) == n_sims
        assert results.dtype == np.float64
    
    def test_simulation_returns_positive_costs(self, engine):
        """Test that simulated costs are positive."""
        results = engine.run_simulation(n_sims=100, time_horizon=1.0)
        
        # All costs should be positive
        assert (results > 0).all()
    
    def test_simulation_reproducibility(self, engine):
        """Test that simulation with same seed produces same results."""
        np.random.seed(42)
        results1 = engine.run_simulation(n_sims=100, time_horizon=1.0)
        
        np.random.seed(42)
        results2 = engine.run_simulation(n_sims=100, time_horizon=1.0)
        
        np.testing.assert_array_almost_equal(results1, results2)


class TestAnalytics:
    """Tests for Analytics module."""
    
    def test_risk_metrics_positive(self, engine, portfolio):
        """Smoke test that risk metrics are positive."""
        # Run simulation
        sim_costs = engine.run_simulation(n_sims=1000, time_horizon=1.0)
        baseline = portfolio.calculate_baseline_cost()
        cost_overruns = sim_costs - baseline
        
        # Calculate risk metrics
        var, es = calculate_risk_metrics(cost_overruns, confidence_level=0.95)
        
        # Both should be positive (more cost = positive overrun)
        assert var > 0
        assert es > 0
        assert es >= var  # ES should be >= VaR
    
    def test_evt_returns_valid_parameters(self, engine, portfolio):
        """Test that EVT fitting returns valid GPD parameters."""
        sim_costs = engine.run_simulation(n_sims=1000, time_horizon=1.0)
        baseline = portfolio.calculate_baseline_cost()
        cost_overruns = sim_costs - baseline
        
        shape, scale, evt_var = fit_evt_gpd(cost_overruns, threshold_quantile=0.90)
        
        # Check that parameters are finite
        assert np.isfinite(shape)
        assert np.isfinite(scale)
        assert np.isfinite(evt_var)
        assert scale > 0  # Scale must be positive
    
    def test_var_increases_with_confidence(self, engine, portfolio):
        """Test that VaR increases with confidence level."""
        sim_costs = engine.run_simulation(n_sims=1000, time_horizon=1.0)
        baseline = portfolio.calculate_baseline_cost()
        cost_overruns = sim_costs - baseline
        
        var_95, _ = calculate_risk_metrics(cost_overruns, 0.95)
        var_99, _ = calculate_risk_metrics(cost_overruns, 0.99)
        
        assert var_99 > var_95


if __name__ == '__main__':
    pytest.main([__file__, '-v'])