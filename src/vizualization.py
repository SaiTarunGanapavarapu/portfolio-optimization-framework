#-------------------------------------------------------------------------------
# Name:         visualization.py
# Purpose:      Class to handle all project visualizations using matplotlib.
#
# Author:       Sai Tarun Ganapavarapu
#
# Created:      02-22-2026
# Licence:      MIT License
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Visualizer:
    """
    Static class to generate plots for portfolio optimization results.
    """
    @staticmethod
    def plotEfficientFrontier(meanReturns, covMatrix, riskFreeRate, optimalPortfolio=None):
        """
        Plots the Markowitz Efficient Frontier based on Monte Carlo simulation.
        """
        print("\n--- Plotting Efficient Frontier ---")
        
        numAssets = len(meanReturns)
        numPortfolios = 10000
        
        # Containers for simulation results
        results = np.zeros((3, numPortfolios))
        
        # Monte Carlo Simulation
        for i in range(numPortfolios):
            weights = np.random.random(numAssets)
            weights /= np.sum(weights)
            
            pReturn = np.sum(meanReturns * weights)
            pStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))
            
            results[0, i] = pStdDev # Volatility (X-axis)
            results[1, i] = pReturn # Return (Y-axis)
            results[2, i] = (pReturn - riskFreeRate) / pStdDev # Sharpe Ratio

        # Convert results to DataFrame
        resultsFrame = pd.DataFrame(results.T, columns=['Volatility', 'Return', 'Sharpe'])

        # Find Max Sharpe Point from simulation
        maxSharpe = resultsFrame.loc[resultsFrame['Sharpe'].idxmax()]
        
        plt.figure(figsize=(12, 6))
        
        # Plot all simulated portfolios
        plt.scatter(resultsFrame.Volatility, resultsFrame.Return, c=resultsFrame.Sharpe, cmap='viridis', s=10, alpha=0.5)
        plt.colorbar(label='Sharpe Ratio')
        
        # Plot Max Sharpe Point from simulation
        plt.scatter(maxSharpe['Volatility'], maxSharpe['Return'], marker='*', color='r', s=100, label='Simulated Max Sharpe')

        # Plot the mathematically optimized portfolio (from MarkowitzOptimizer result)
        if optimalPortfolio:
            plt.scatter(optimalPortfolio['volatility'], optimalPortfolio['return'], marker='X', color='orange', s=100, label='Optimized Markowitz MV')

        plt.title('Markowitz Efficient Frontier')
        plt.xlabel('Annualized Volatility (Standard Deviation)')
        plt.ylabel('Annualized Return')
        plt.legend(labelspacing=0.8)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    @staticmethod
    def plotEvaluationResults(portfolioValue, title="Evaluation Performance"):
        """
        Plots the cumulative value of a single out-of-sample evaluated portfolio over time.
        """
        print(f"\n--- Plotting {title} ---")
        plt.figure(figsize=(12, 6))
        plt.plot(portfolioValue.index, portfolioValue.values, label='Portfolio Value', color='darkblue')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (USD)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

    