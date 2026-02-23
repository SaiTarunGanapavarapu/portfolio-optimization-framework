#-------------------------------------------------------------------------------
# Name:        metrics.py
# Purpose:     Computes returns and variance of all stocks and portfolio.
#
# Author:      Sai Tarun Ganapavarapu
#
# Created:     02-18-2026
# Licence:     MIT License
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd

class MetricsCalculator:
    """
    A class to calculate portfolio performance metrics.
    """
    def __init__(self, data, annualizationFactor = 252):
        self.data = data
        self.annualizationFactor = annualizationFactor
        self.meanReturns = None
        self.meanReturnsArithmetic = None
        self.meanReturnsGeometric = None
        self.covMatrix = None
        self.calculateMetrics()

    def calculateMetrics(self):
        """
        Calculates annualized mean returns and the annualized covariance matrix.
        """
        if self.data is None or self.data.empty:
            return None, None
        
        returns = self.data.pct_change().dropna()
        # Arithmetic (simple) annualized mean
        self.meanReturnsArithmetic = returns.mean() * self.annualizationFactor

        # Geometric (compound) annualized mean: use log returns for numerical stability
        logMeans = np.log1p(returns).mean()
        dailyGeo = np.exp(logMeans) - 1
        self.meanReturnsGeometric = (1 + dailyGeo) ** self.annualizationFactor - 1

        # Backwards-compatible alias (default to arithmetic)
        self.meanReturns = self.meanReturnsArithmetic

        # Covariance annualized (sample covariance of daily returns scaled by periods as covariance scales linearly with time)
        self.covMatrix = returns.cov() * self.annualizationFactor

    def portfolioPerformance(self, weights):
        """
        Calculates the expected return and volatility of a portfolio.
        
        Args:
            weights (np.array): Portfolio weights.
            
        Returns:
            tuple: (portfolioReturn, portfolioStdDev)
        """
        portfolioReturn = np.sum(self.meanReturns * weights)
        portfolioStdDev = np.sqrt(np.dot(weights.T, np.dot(self.covMatrix, weights)))
        
        return portfolioReturn, portfolioStdDev

    def getMeanReturns(self, method='arithmetic'):
        """
        Return mean returns according to specified method.

        method: 'arithmetic' (default) or 'geometric'
        Returns a pandas Series with the same index as the computed values.
        """
        if method in ('arithmetic', 'arith'):
            # return a copy to avoid accidental in-place modification by callers
            return self.meanReturnsArithmetic.copy()
        elif method in ('geometric', 'geo'):
            return self.meanReturnsGeometric.copy()
        else:
            raise ValueError(f"Unknown mean method: {method}. Use 'arithmetic' or 'geometric'.")
