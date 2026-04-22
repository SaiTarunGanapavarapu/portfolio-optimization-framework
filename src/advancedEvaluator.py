#-------------------------------------------------------------------------------
# Name:        advancedEvaluator.py
# Purpose:     Walk-forward backtester supporting all Phase-2 optimisers,
#              robust covariance estimators, and the advanced cost model.
#
# Author:      Sai Tarun Ganapavarapu
#
# Created:     03-31-2026
# Licence:     MIT License
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import logging

from .advancedOptimizer import AdvancedOptimizer
from .advancedMetrics import fullTearsheet
from .transactionCosts import TransactionCostModel
from .robustCovariance import RobustCovarianceEstimator
from .dataLoader import DataLoader

logger = logging.getLogger(__name__)


class AdvancedEvaluator:
    """
    Walk-forward backtester that integrates:
    - Any AdvancedOptimizer strategy (or static weights)
    - Any RobustCovarianceEstimator
    - TransactionCostModel
    - Full tearsheet via advancedMetrics

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices indexed by date.
    optimizer : AdvancedOptimizer or None
        If None, static equal-weight allocation is used.
    covEstimator : RobustCovarianceEstimator or None
        If None, sample covariance of the training window is used.
    costModel : TransactionCostModel
        Transaction cost model applied at each rebalance.
    rebalancingPeriod : str
        Pandas offset alias for rebalancing frequency ('Q', 'M', 'Y').
    trainWindow : int
        Number of trading days in the look-back training window.
    riskFreeRate : float
        Annualised risk-free rate used in performance metrics.
    benchmarkTicker : str
        Ticker symbol for the benchmark (default 'SPY').
    initialCapital : float
        Starting portfolio value in dollars.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        optimizer: AdvancedOptimizer | None,
        covEstimator: RobustCovarianceEstimator | None,
        costModel: TransactionCostModel,
        rebalancingPeriod: str = 'Q',
        trainWindow: int = 252,
        riskFreeRate: float = 0.04,
        benchmarkTicker: str = 'SPY',
        initialCapital: float = 100_000.0,
    ):
        if prices is None or prices.empty:
            raise ValueError("prices must be a non-empty DataFrame")
        self.prices = prices
        self.optimizer = optimizer
        self.covEstimator = covEstimator
        self.costModel = costModel
        self.rebalancingPeriod = rebalancingPeriod
        self.trainWindow = trainWindow
        self.riskFreeRate = riskFreeRate
        self.benchmarkTicker = benchmarkTicker
        self.initialCapital = initialCapital

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def runWalkForward(self, startDate: str, endDate: str) -> dict:
        """
        Run the walk-forward backtest over [startDate, endDate].

        Parameters
        ----------
        startDate : str
            ISO date string for backtest start (must be within self.prices).
        endDate : str
            ISO date string for backtest end (must be within self.prices).

        Returns
        -------
        dict
            Keys:
            - 'portfolioValues'  : pd.Series of daily NAV
            - 'benchmarkValues'  : pd.Series of benchmark NAV (or None)
            - 'weightsHistory'   : pd.DataFrame (rebalance date × assets)
            - 'regimeHistory'    : pd.Series of regime labels at each rebalance
            - 'turnoverHistory'  : pd.Series of turnover at each rebalance
            - 'costHistory'      : pd.Series of dollar costs at each rebalance
            - 'metrics'          : dict from fullTearsheet
        """
        pricesSlice = self.prices.loc[startDate:endDate]
        if pricesSlice.empty:
            raise ValueError(f"No price data found between {startDate} and {endDate}")

        allDates = pricesSlice.index
        dailyReturns = pricesSlice.pct_change().fillna(0.0)
        N = pricesSlice.shape[1]

        freqMap = {'Y': 'YE', 'Q': 'QE', 'M': 'ME'}
        freq = freqMap.get(self.rebalancingPeriod, self.rebalancingPeriod)
        rebalanceDates = pd.date_range(
            start=allDates[0], end=allDates[-1], freq=freq
        ).intersection(allDates)

        portfolioValues = pd.Series(index=allDates, dtype=float)
        weightsHistory: dict[pd.Timestamp, np.ndarray] = {}
        regimeHistory: dict[pd.Timestamp, str] = {}
        turnoverHistory: dict[pd.Timestamp, float] = {}
        costHistory: dict[pd.Timestamp, float] = {}

        currentCapital = float(self.initialCapital)
        currentWeights = np.full(N, 1.0 / N)
        portfolioValues.iloc[0] = currentCapital

        for i, date in enumerate(allDates[1:], 1):
            if date in rebalanceDates:
                startIdx = max(0, i - self.trainWindow)
                trainPrices = self.prices.iloc[
                    self.prices.index.get_loc(allDates[startIdx]):
                    self.prices.index.get_loc(allDates[i - 1]) + 1
                ]

                newWeights, regime = self._computeWeights(trainPrices)

                turnover = float(np.sum(np.abs(newWeights - currentWeights)))
                cost = self.costModel.computeCost(currentWeights, newWeights, currentCapital)
                currentCapital -= cost
                currentCapital = max(currentCapital, 0.0)

                weightsHistory[date] = newWeights
                regimeHistory[date] = regime
                turnoverHistory[date] = turnover
                costHistory[date] = cost

                currentWeights = newWeights

            dayReturn = float(np.dot(currentWeights, dailyReturns.iloc[i]))
            currentCapital *= (1.0 + dayReturn)
            portfolioValues.iloc[i] = currentCapital

        portfolioValues = portfolioValues.dropna()

        whDf = (
            pd.DataFrame(weightsHistory, index=pricesSlice.columns).T
            if weightsHistory else pd.DataFrame()
        )
        whDf.index.name = 'date'

        regimeSeries = pd.Series(regimeHistory, name='regime')
        turnoverSeries = pd.Series(turnoverHistory, name='turnover')
        costSeries = pd.Series(costHistory, name='cost')

        benchmarkValues = self._fetchBenchmark(startDate, endDate)

        metrics = fullTearsheet(
            portfolioValues,
            benchmarkValues=benchmarkValues,
            riskFreeRate=self.riskFreeRate,
        )

        return {
            'portfolioValues': portfolioValues,
            'benchmarkValues': benchmarkValues,
            'weightsHistory': whDf,
            'regimeHistory': regimeSeries,
            'turnoverHistory': turnoverSeries,
            'costHistory': costSeries,
            'metrics': metrics,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _computeWeights(self, trainPrices: pd.DataFrame) -> tuple[np.ndarray, str]:
        """Compute target weights and regime label from training prices."""
        N = trainPrices.shape[1]
        trainReturns = trainPrices.pct_change().dropna()

        if len(trainReturns) < 2:
            logger.warning("Insufficient training data; using equal weights.")
            return np.full(N, 1.0 / N), 'normal'

        regime = 'normal'
        if self.covEstimator is not None:
            if self.covEstimator.method == 'regimeAware':
                covMatrix, regime = self.covEstimator.estimateWithRegime(trainReturns)
            else:
                covMatrix = self.covEstimator.estimate(trainReturns)
        else:
            covMatrix = trainReturns.cov().values * 252

        covMatrix = _ensurePsd(covMatrix)
        meanReturns = trainReturns.mean().values * 252

        if self.optimizer is None:
            return np.full(N, 1.0 / N), regime

        try:
            weights = self.optimizer.optimize(
                returns=trainReturns,
                covMatrix=covMatrix,
                meanReturns=meanReturns,
            )
        except Exception as exc:
            logger.warning("Optimizer raised %s; using equal weights.", exc)
            weights = np.full(N, 1.0 / N)

        return weights, regime

    def _fetchBenchmark(self, startDate: str, endDate: str) -> pd.Series | None:
        """Download benchmark prices and return a NAV series."""
        try:
            benchPrices = DataLoader.getData(self.benchmarkTicker, startDate, endDate)
            if benchPrices is None or benchPrices.empty:
                return None
            bv = benchPrices.iloc[:, 0]
            return bv / bv.iloc[0] * self.initialCapital
        except Exception as exc:
            logger.warning("Could not fetch benchmark '%s': %s", self.benchmarkTicker, exc)
            return None


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def _ensurePsd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Clip negative eigenvalues to guarantee positive semi-definiteness."""
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, epsilon)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
