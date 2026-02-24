#-------------------------------------------------------------------------------
# Name:          evaluator.py
# Purpose:       A class to run an out-of-sample evaluation and report performance.
#
# Author:        Sai Tarun Ganapavarapu
#
# Created:       02-19-2026
# Licence:       MIT License
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np

class OutOfSampleEvaluator:
    """
    A class to run an out-of-sample evaluation and produce performance metrics.
    """
    def __init__(self, data, initialCapital=100000, riskFreeRate=0.0, transactionCostRate=0.0):
        self.data = data
        self.initialCapital = initialCapital
        self.tickers = data.columns
        self.numAssets = len(self.tickers)
        self.riskFreeRate = riskFreeRate
        self.transactionCostRate = transactionCostRate  # as decimal ( 0.001 = 0.1%)

    def runEvaluation(self, rebalanceFunction, initialTrainingPeriod, rebalancingPeriod):
        """
        Runs a rebalancing evaluation using a specified rebalance function.
        The rebalanceFunction is expected to return a full weights array for the
        current set of tickers. 
        
        Rebalancing occurs according to rebalancingPeriod (e.g., 'M' monthly, 'Q' quarterly, 'Y' yearly).
        Transaction costs are applied as a percentage of the traded capital whenever rebalancing occurs.
        
        Returns (portfolioValueSeries, weightsHistoryDataFrame).
        """
        
        # Initialize portfolio value and weights history
        portfolioValue = pd.Series(index=self.data.index, dtype=float)
        weightsHistory = pd.DataFrame(index=self.data.index, columns=self.tickers, dtype=float)
        transactionCosts = pd.Series(index=self.data.index, dtype=float)

        # Force initial rebalance using the first trading day
        initialTrainEnd = self.data.index[0]
        trainData = self.data.loc[:initialTrainEnd]
        weights = rebalanceFunction(trainData)
        previousWeights = np.zeros(len(weights))

        currentCapital = self.initialCapital
        
        # Initial transaction costs (buying initial positions)
        initialTradedFraction = np.sum(np.abs(weights - previousWeights))
        initialCost = currentCapital * initialTradedFraction * self.transactionCostRate
        currentCapital -= initialCost
        transactionCosts.iloc[0] = initialCost
        
        portfolioValue.iloc[0] = currentCapital
        weightsHistory.iloc[0] = weights
        previousWeights = weights.copy()

        # Daily returns of assets
        dailyReturns = self.data.pct_change().fillna(0)

        # Align rebalance dates with actual trading dates
        freqMap = {'Y': 'YE', 'Q': 'QE', 'M': 'ME'}
        freq = freqMap.get(rebalancingPeriod, rebalancingPeriod)
        
        rebalanceDates = pd.date_range(
            start=self.data.index[0],
            end=self.data.index[-1],
            freq=freq
        ).intersection(self.data.index)

        lastRebalanceDate = self.data.index[0]

        for i, date in enumerate(self.data.index[1:], 1):
            # Check if rebalancing should occur
            needsRebalance = date >= lastRebalanceDate and date in rebalanceDates and i > 0
            
            if needsRebalance:
                trainData = self.data.loc[:self.data.index[i-1]]   # prevents look-ahead bias.
                weights = rebalanceFunction(trainData)
                
                # Calculate transaction costs based on weight changes
                tradedFraction = np.sum(np.abs(weights - previousWeights))
                transactionCost = currentCapital * tradedFraction * self.transactionCostRate
                currentCapital -= transactionCost
                transactionCosts.iloc[i] = transactionCost
                
                previousWeights = weights.copy()
                lastRebalanceDate = date
            else:
                transactionCosts.iloc[i] = 0.0

            # Apply daily returns
            dailyReturn = np.sum(weights * dailyReturns.loc[date])
            currentCapital *= (1 + dailyReturn)

            portfolioValue.iloc[i] = currentCapital
            weightsHistory.iloc[i] = weights

        # Store transaction costs
        self.lastTransactionCosts = transactionCosts
        
        return portfolioValue.dropna(), weightsHistory.dropna()

    def generateReport(self, portfolioValue):
        """
        Generates a performance report for the evaluated portfolio.
        Returns a dict with totalReturn, annualizedReturn, annualizedVolatility,
        sharpeRatio, and maxDrawdown.
        """
        totalReturn = (portfolioValue.iloc[-1] / portfolioValue.iloc[0]) - 1
        dailyReturns = portfolioValue.pct_change().dropna()

        annualizationFactor = 252
        annualizedReturn = (1 + dailyReturns.mean())**annualizationFactor - 1   # yearly compounded return
        annualizedVolatility = dailyReturns.std() * np.sqrt(annualizationFactor)

        sharpeRatio = np.nan if annualizedVolatility == 0 else \
            (annualizedReturn - self.riskFreeRate) / annualizedVolatility

        peak = portfolioValue.expanding(min_periods=1).max()
        drawdown = (portfolioValue - peak) / peak
        maxDrawdown = drawdown.min()

        return {
            'totalReturn': totalReturn,
            'annualizedReturn': annualizedReturn,
            'annualizedVolatility': annualizedVolatility,
            'sharpeRatio': sharpeRatio,
            'maxDrawdown': maxDrawdown
        }
