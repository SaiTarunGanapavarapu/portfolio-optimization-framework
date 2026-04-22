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

# Phase 2 imports
from src.robustCovariance import RobustCovarianceEstimator
from src.advancedOptimizer import AdvancedOptimizer
from src.transactionCosts import TransactionCostModel
from src.advancedEvaluator import AdvancedEvaluator
from src.advancedMetrics import fullTearsheet


class PortfolioEngine:
    """
    Orchestrator class to coordinate data, metrics, forecasting, optimization, and evaluation.
    """
    def __init__(self, tickers, startDate, endDate, splitDate, riskFreeRate,
                 meanMethod, shrinkage, rebalancingPeriod,
                 transactionCostRate, initialCapital,
                 # Phase 2 options
                 covarianceMethod='ewma', optimizationMethod='minCvar',
                 maxWeight=1.0, minWeight=0.0, turnoverPenalty=0.0, cvarAlpha=0.05):

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
        # Phase 2 options
        self.covarianceMethod = covarianceMethod      # 'ewma', 'denoised', 'regimeAware'
        self.optimizationMethod = optimizationMethod  # 'minCvar', 'riskParity', 'maxDiversification'
        self.maxWeight = maxWeight
        self.minWeight = minWeight
        self.turnoverPenalty = turnoverPenalty
        self.cvarAlpha = cvarAlpha

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

    def runPhase2(self):
        """
        Strategy 2: Advanced optimization using robust covariance + CVaR/risk-parity/max-diversification.
        Runs a walk-forward backtest over the out-of-sample period.
        """
        print("\n--- Phase 2: Robust covariance + CVaR/risk-parity/max-diversification Optimization ---")

        if self.data is None:
            self.data = DataLoader.getData(self.tickers, self.startDate, self.endDate)
        if self.data is None:
            print("Error: could not load data for Phase 2.")
            return

        # Only use the test period for the walk-forward
        testData = self.data.loc[self.data.index >= self.splitDate]
        if testData.empty:
            print("Error: no out-of-sample data after splitDate.")
            return

        # Build Phase 2 components
        covEstimator = RobustCovarianceEstimator(method=self.covarianceMethod)
        optimizer = AdvancedOptimizer(
            method=self.optimizationMethod,
            riskFreeRate=self.riskFreeRate,
            maxWeight=self.maxWeight,
            minWeight=self.minWeight,
            turnoverPenalty=self.turnoverPenalty,
            cvarAlpha=self.cvarAlpha,
        )
        costModel = TransactionCostModel(
            commissionRate=self.transactionCostRate,
            spreadCost=self.transactionCostRate / 2,
        )
        evaluator = AdvancedEvaluator(
            prices=self.data,
            optimizer=optimizer,
            covEstimator=covEstimator,
            costModel=costModel,
            rebalancingPeriod=self.rebalancingPeriod,
            trainWindow=252,
            riskFreeRate=self.riskFreeRate,
            initialCapital=self.initialCapital,
        )

        results = evaluator.runWalkForward(
            startDate=str(testData.index[0].date()),
            endDate=str(testData.index[-1].date()),
        )

        portfolioValues = results['portfolioValues']
        metrics = results['metrics']
        strategyName = f"Phase 2: {self.optimizationMethod} ({self.covarianceMethod})"
        self.allEvaluationResults[strategyName] = portfolioValues

        # Print the full portfolio weights chosen in second phase
        # Print final Phase 2 portfolio weights (last rebalance)
        weightsHistory = results.get('weightsHistory')
        if weightsHistory is not None and not weightsHistory.empty:
            latestDate = weightsHistory.index[-1]
            latestWeights = weightsHistory.iloc[-1]

            print(f"\nOptimal Phase 2 Weights (Last Rebalance: {latestDate.date()}):")
            for ticker, weight in latestWeights.items():
                print(f"  {ticker}: {weight:.2%}")
            print(f"Verification: Total Weights Sum = {latestWeights.sum():.4f}")
        else:
            print("\nNo Phase 2 rebalance weights available.")


        print(f"\n--- {strategyName} Out-of-Sample Report ---")
        print(f"Total Return:          {metrics.get('totalReturn', float('nan')) * 100:.2f}%")
        print(f"CAGR:                  {metrics.get('cagr', float('nan')) * 100:.2f}%")
        print(f"Annualized Volatility: {metrics.get('volatility', float('nan')) * 100:.2f}%")
        print(f"Sharpe Ratio:          {metrics.get('sharpe', float('nan')):.2f}")
        print(f"Sortino Ratio:         {metrics.get('sortino', float('nan')):.2f}")
        print(f"Calmar Ratio:          {metrics.get('calmar', float('nan')):.2f}")
        print(f"Max Drawdown:          {metrics.get('maxDrawdown', float('nan')) * 100:.2f}%")
        print(f"CVaR 95%:              {metrics.get('cvar95', float('nan')) * 100:.2f}%")

        regimeHistory = results['regimeHistory']
        if regimeHistory is not None and not regimeHistory.empty:
            stressedPct = (regimeHistory == 'stressed').mean() * 100
            print(f"Stressed regime:       {stressedPct:.1f}% of rebalance dates")

        # Visualization
        Visualizer.plotEvaluationResults(portfolioValues, title=f"Phase 2: {strategyName} (Out-of-Sample)")
        Visualizer.plotComparison(self.allEvaluationResults, title="Phase 2: Strategy Comparison (Out-of-Sample)")

    def runAnalysis(self):
        """
        Orchestrates all implemented phases and visualizes the performance.
        """
        self.runPhase0()
        self.runPhase1()
        self.runPhase2()

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
        # Phase 2 options
        covarianceMethod = 'ewma',  # 'ewma', 'denoised', 'regimeAware'
        optimizationMethod = 'minCvar',     # 'minCvar', 'riskParity', 'maxDiversification'
        maxWeight = 0.3,                    # Max weight per asset (30%)    
        minWeight = 0.0,                    # Min weight per asset (0%)
        turnoverPenalty = 0.001,             # 0.1% turnover penalty (as decimal)
        cvarAlpha = 0.05                    # CVaR confidence level (5%)
    )

    portfolio.runAnalysis()

