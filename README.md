# Nissan Commodity Risk Engine

A quantitative risk modeling framework designed to simulate and hedge commodity price exposures for automotive manufacturing.

##  Overview

This project quantifies the financial risk exposure of a major automotive manufacturer (Nissan Motor Corp.) to volatility in raw material inputs (Steel, Aluminum, Copper, etc.). 

It implements a **Monte Carlo simulation engine** using **Correlated Geometric Brownian Motion (GBM)** to forecast procurement cost distributions over a 1-year horizon. It further utilizes **Extreme Value Theory (EVT)** to model tail risks that standard normal distributions fail to capture.

### Key Features
* **Correlated Asset Simulation:** Uses Cholesky Decomposition to preserve the statistical relationships between commodity prices (e.g., Steel and Aluminum correlation).
* **Vectorized Performance:** Core simulation engine is fully vectorized using NumPy for high-performance computation (10,000+ paths).
* **Advanced Risk Metrics:** Calculates Value at Risk (VaR) and Expected Shortfall (ES) at multiple confidence intervals.
* **Tail Risk Modeling:** Fits a Generalized Pareto Distribution (GPD) via EVT to estimate 1-in-200 year loss events (99.5% VaR).
* **Hedging Strategy:** Models the P&L impact of a 60% strategic hedging program on key commodities.

##  Project Structure

The codebase is organized as a modular Python package:

```text
├── src/
│   ├── portfolio.py      # Commodity and Portfolio object definitions
│   ├── simulation.py     # Vectorized Monte Carlo engine (GBM + Cholesky)
│   └── analytics.py      # Statistical libraries (VaR, ES, EVT/GPD fitting)
├── data/
│   ├── nissan_exposure.csv  # Portfolio volume and volatility definitions
│   └── correlation.csv      # Correlation matrix for commodities
├── tests/
│   └── test_simulation.py   # Pytest suite for logic verification
└── risk_analysis_demo.py    # CLI entry point for running the analysis
