#-------------------------------------------------------------------------------
# Name:         main.py
# Purpose:      Main orchestrator/engine for the portfolio optimization workflow.
#
# Author:       Sai Tarun Ganapavarapu
#
# Created:      02-23-2026
# Licence:      MIT License
#-------------------------------------------------------------------------------

import pandas as pd
import numpy as np

# Import classes from the src directory using absolute imports
from src.dataLoader import DataLoader
from src.metrics import MetricsCalculator
from src.markowitzOptimizer import MarkowitzOptimizer
from src.evaluator import OutOfSampleEvaluator
from src.vizualization import Visualizer


class PortfolioEngine:
    """
    Orchestrator class to coordinate data, metrics, forecasting, optimization, and evaluation.
    """
    def __init__(self, tickers, startDate, endDate, splitDate, riskFreeRate, 
                 meanMethod, shrinkage, rebalancingPeriod, 
                 transactionCostRate, initialCapital):

        self.tickers = tickers
        self.startDate = startDate
        self.endDate = endDate
        self.splitDate = splitDate
        self.riskFreeRate = riskFreeRate
        self.initialCapital = initialCapital
        self.data = None
        self.targetReturn = None
        self.allEvaluationResults = {}
        # Markowitz options
        self.meanMethod = meanMethod  # 'arithmetic' or 'geometric'
        self.shrinkage = shrinkage    # 'ledoit' or float in [0,1]
        # Rebalancing and transaction cost options
        self.rebalancingPeriod = rebalancingPeriod  # 'M', 'Q', 'Y', '10Y'
        self.transactionCostRate = transactionCostRate  # as decimal

    def _splitData(self, splitDate = None):
        """
        Splits dataset into training (before splitDate) and testing (after splitDate).
        """
        if self.data is None:
            self.data = DataLoader.getData(self.tickers, self.startDate, self.endDate)
        
        if self.data is None: return None, None

        trainingData = self.data.loc[self.data.index < splitDate]
        testingData  = self.data.loc[self.data.index >= splitDate]

        if trainingData.empty or testingData.empty:
            print("Error: Training or testing dataset is empty. Check your dates.")
            return None, None

        print(f"\nTraining period: {trainingData.index.min().strftime('%Y-%m-%d')} to {trainingData.index.max().strftime('%Y-%m-%d')}")
        print(f"Testing period: {testingData.index.min().strftime('%Y-%m-%d')} to {testingData.index.max().strftime('%Y-%m-%d')}")
        return trainingData, testingData

    def _evaluateStrategy(self, weights, testingData, optimizedTickers, strategyName):
        """
        Runs evaluation of a fixed strategy on testing data.
        Returns the portfolio value series for visualization.
        """
        evaluator = OutOfSampleEvaluator(testingData, initialCapital = self.initialCapital, riskFreeRate = self.riskFreeRate, 
                                          transactionCostRate = self.transactionCostRate)

        # Define a rebalancing function that simply returns the fixed, optimized weights
        def rebalanceStatic(_):
            # Create a full weight array ordered by the test data columns
            weightsDict = dict(zip(optimizedTickers, weights))
            fullWeights = np.array([weightsDict.get(ticker, 0.0) for ticker in testingData.columns])
            return fullWeights

        # Use the configured rebalancing period
        # `runEvaluation` returns (portfolioSeries, weightsHistory) -> unpack both
        portfolioValues, weightsHistory = evaluator.runEvaluation(rebalanceStatic, initialTrainingPeriod='1D', 
                                                                    rebalancingPeriod = self.rebalancingPeriod)

        if portfolioValues is not None and not getattr(portfolioValues, 'empty', False):
            report = evaluator.generateReport(portfolioValues)
            self.allEvaluationResults[strategyName] = portfolioValues

            print(f"\n--- {strategyName} Out-of-Sample Report ---")
            print(f"Total Return: {report['totalReturn'] * 100:.2f}%")
            print(f"Annualized Return: {report['annualizedReturn'] * 100:.2f}%")
            print(f"Annualized Volatility: {report['annualizedVolatility'] * 100:.2f}%")
            print(f"Sharpe Ratio: {report['sharpeRatio']:.2f}")
            print(f"Max Drawdown: {report['maxDrawdown'] * 100:.2f}%")
            
            return portfolioValues
        else:
            print(f"{strategyName} evaluation failed or returned empty results.")
            return None

    def runPhase0(self):
        """
        Strategy 0: Naive Equal-Weight Benchmark.
        """
        print("\n--- Phase 0: Equal-Weight Benchmark (Naive) ---")
        _, testingData = self._splitData(self.splitDate)
        
        # 1/N weights
        numAssets = len(self.tickers)
        eqWeights = np.full(numAssets, 1.0 / numAssets)
        
        # Use your existing evaluation logic
        portfolioValues = self._evaluateStrategy(
            eqWeights, 
            testingData, 
            self.tickers, 
            "Equal-Weight (Benchmark)"
        )        

    def runPhase1(self):
        """
        Strategy 1: Markowitz Mean-Variance Optimization.
        """
        print("\n--- Phase 1: Markowitz Mean-Variance Optimization (Historical) ---")
        trainingData, testingData = self._splitData(self.splitDate)
        if trainingData is None: return

        metrics = MetricsCalculator(trainingData)
        optimizer = MarkowitzOptimizer(metrics=metrics, meanMethod = self.meanMethod, shrinkage = self.shrinkage)
        results = optimizer.optimizePortfolio(riskFreeRate = self.riskFreeRate)
 
        if results:
            self.targetReturn = results['return']
            meanSeries = metrics.getMeanReturns(method = self.meanMethod)
            optimizedTickers = meanSeries.index.tolist()
            
            print(f"Optimal Markowitz Weights (Trained on historical data):")
            for ticker, weight in zip(optimizedTickers, results['weights']):
                print(f"  {ticker}: {weight:.2%}")
            print(f"Verification: Total Weights Sum = {np.sum(results['weights']):.4f}")
            
            # --- Visualization 1: Efficient Frontier ---
            Visualizer.plotEfficientFrontier(meanSeries, metrics.covMatrix, self.riskFreeRate, optimalPortfolio = results)

            # Run out-of-sample evaluation
            portfolioValues = self._evaluateStrategy(results['weights'], testingData, optimizedTickers, "Markowitz MV")
            
            # --- Visualization 2: Evaluation Performance ---
            if portfolioValues is not None:
                Visualizer.plotEvaluationResults(portfolioValues, title="Phase 1: Markowitz MV Evaluation (Out-of-Sample)")

            # --- Visualization 3: Strategy Comparison ---
            if portfolioValues is not None:
                Visualizer.plotComparison(self.allEvaluationResults, title="Phase 1: Strategy Comparison (Out-of-Sample)")

    def runAnalysis(self):
        """
        Orchestrates phase 1 and visualizes the performance. 
        """
        self.runPhase0()
        self.runPhase1()

if __name__ == '__main__':

    # --- Project Execution ---
    portfolio = PortfolioEngine(
        tickers = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'TSM', 'BRK-B'],
        initialCapital = 1000,
        startDate = '2015-01-01',
        endDate = '2026-02-20',
        splitDate = '2020-01-01',
        riskFreeRate = 0.04,  # Example risk-free rate
        meanMethod = 'arithmetic',  # 'arithmetic' or 'geometric'
        shrinkage = 'ledoit',
        rebalancingPeriod = 'Y',      # 'M' = monthly, 'Q' = quarterly, 'Y' = yearly
        transactionCostRate = 0.001,  # 0.1% transaction cost (as decimal)

    )

    portfolio.runAnalysis()

