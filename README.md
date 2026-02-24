# Portfolio Optimization Project

A modular, multi-phase quantitative finance framework for portfolio construction and backtesting. This project implements classic Markowitz theory, advanced time-series risk forecasting, and machine learning-driven return prediction.

## Overview

This repository implements a pipeline for:

1. **Data ingestion**: Fetching and cleaning adjusted close prices.
2. **Estimation**: Calculating mean returns and covariance matrices (implementing **Ledoit-Wolf** shrinkage)
3. **Optimization**: Solving for Mean-Variance Efficient portfolios with convex optimization .
4. **Walk-forward evaluation**: Out-of-sample backtesting with configurable rebalancing schedules and transaction cost modeling.

The codebase is structured to separate optimization logic from data ingestion and evaluation, facilitating independent testing and modular expansion.

## Key Features

### Phase 0

- **Equal-Weight Benchmark**: Builds a naive baseline portfolio with 1/N allocation across selected assets.
- **Out-of-Sample Evaluation**: Uses the same evaluator, rebalancing schedule, and transaction cost model as optimized strategies for fair comparison.
- **Strategy Comparison Ready**: Stores benchmark results so they can be compared directly against Markowitz outputs.

### Phase 1

- **Covariance Estimation**: Includes Ledoit-Wolf shrinkage (via scikit-learn) to improve conditioning of the covariance matrix for high-dimensional datasets.
- **Convex Optimization**: Uses scipy.optimize (SLSQP) to maximize the Sharpe Ratio, subject to constraints (e.g., long-only, full investment).
- **Regularization**: Applies Tikhonov regularization (**Σ+λI**) to ensure positive definiteness during optimization.
- **Walk-Forward Evaluation**: Prevents look-ahead bias by testing weights on unseen data.
  - **Rebalancing**: Supports Monthly, Quarterly, and Custom frequencies.
  - **Transaction Costs**: Models slippage and commissions as a linear function of portfolio turnover.

## Project Structure

```text
src/
├── dataLoader.py          # yfinance wrapper & data cleaning
├── metrics.py             # Statistical moment estimation (Mean/Cov)
├── markowitzOptimizer.py  # Convex optimization (Sharpe/MVO)
├── evaluator.py           # Walk-forward out-of-sample backtester
└── vizualization.py       # Efficient frontier & performance plots
main.py                    # Main orchestrator (Configurable entry point)
main.ipynb                 # Interactive research notebook
requirements.txt           # Dependencies
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage Example

You can run the full pipeline directly via the central engine:

```Python
from main import PortfolioEngine

# 1. Initialize the framework
portfolio = PortfolioEngine(
    tickers=['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'TSLA'],
    startDate='2015-01-01',
    endDate='2024-01-01',
    splitDate='2023-01-01',
    riskFreeRate = 0.04,  
    meanMethod = 'arithmetic',
    shrinkage = 'ledoit',
    rebalancingPeriod='Q',        # Quarterly rebalancing
    transactionCostRate=0.001     # 0.1% costs
)

# 2. Run Analysis (Executes Phase 0 benchmark + Phase 1 optimization)
portfolio.runAnalysis()
```

### Configuration Options

You can toggle advanced features directly in the constructor:

- **tickers**: List of ticker symbols (e.g., ```['AAPL', 'MSFT]```)
- **riskFreeRate**: Annualized risk-free rate used for Sharpe Ratio calculation.
- **meanMethod**: Method for expected return estimation.
- **shrinkage**: Covariance shrinkage method. Use ```'ledoit'``` for automatic Ledoit-Wolf or a float ```0.5``` for fixed shrinkage.
- **rebalancingPeriod**: Frequency of portfolio rebalancing. Options: ```'M'``` (monthly), ```'Q'``` (quarterly), ```'Y'``` (yearly), and ```'10Y'``` (buy and hold).
- **transactionCostRate**: Proportional transaction cost (e.g., ```0.001``` = 10bps). Applied to total turnover.

## Road Map

- **Phase 0**: Equal-Weight Benchmark and Out-of-Sample Baseline (Completed)
- **Phase 1**: Core Mean-Variance Optimization and Backtesting (Completed)
- **Phase 2**: GARCH-based Volatility Forecasting and CVaR Optimization
- **Phase 3**: Machine Learning Return Prediction (XGBoost/LSTM)
