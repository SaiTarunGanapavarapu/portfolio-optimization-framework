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

    @staticmethod
    def plotComparison(evaluationResults, title="Strategy Comparison (Out-of-Sample)"):
        """
        Generates a bar chart comparing key metrics across different phases/strategies.
        evaluationResults: A dict where keys are strategy names and values are portfolioValue series.
        """
        print(f"\n--- Plotting {title} ---")

        riskFreeRate = 0.04
        metricsRows = []

        for strategyName, portfolioValues in evaluationResults.items():
            dailyReturns = portfolioValues.pct_change().dropna()
            annualizedReturn = (1 + dailyReturns.mean()) ** 252 - 1
            annualizedVolatility = dailyReturns.std() * np.sqrt(252)

            if annualizedVolatility == 0:
                sharpeRatio = np.nan
            else:
                sharpeRatio = (annualizedReturn - riskFreeRate) / annualizedVolatility

            runningPeak = portfolioValues.expanding(min_periods=1).max()
            maxDrawdown = ((portfolioValues - runningPeak) / runningPeak).min()

            metricsRows.append({
                "Strategy": strategyName,
                "Annualized Return (%)": annualizedReturn * 100,
                "Annualized Volatility (%)": annualizedVolatility * 100,
                "Max Drawdown (%)": maxDrawdown * 100,
                "Sharpe Ratio": sharpeRatio
            })

        comparisonFrame = pd.DataFrame(metricsRows).set_index("Strategy").T

        ax = comparisonFrame.plot(kind='bar', figsize=(12, 7), zorder=3)
        plt.title(title, fontsize=15, fontweight='bold')
        plt.ylabel("Value", fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        plt.legend(loc='best', frameon=True, shadow=True)

        for bar in ax.patches:
            yValue = bar.get_height()
            if np.isfinite(yValue):
                ax.annotate(
                    f"{yValue:.2f}",
                    (bar.get_x() + bar.get_width() / 2.0, yValue),
                    ha='center',
                    va='center',
                    xytext=(0, 9),
                    textcoords='offset points'
                )

        plt.tight_layout()
        plt.show()

    