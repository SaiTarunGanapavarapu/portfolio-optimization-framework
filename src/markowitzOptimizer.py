#-------------------------------------------------------------------------------
# Name:        markowitzOptimizer.py
# Purpose:     Class to perfom Markowitx Optimization.
#
# Author:      Sai Tarun Ganapavarapu
#
# Created:     09-23-2025
# Licence:     MIT License
#-------------------------------------------------------------------------------
import numpy as np
from scipy.optimize import minimize
from .metrics import MetricsCalculator
from sklearn.covariance import LedoitWolf

class MarkowitzOptimizer:
    """
    A class to perform Markowitz Mean-Variance Optimization.
    """
    def __init__(self, meanReturns=None, covMatrix=None, metrics: MetricsCalculator = None, meanMethod='arithmetic', shrinkage='ledoit', regularization=1e-8):
        """
        Initialize the optimizer.

        regularization: small diagonal regularizer added to covariance to improve numeric stability.
        """
        if metrics is not None:
            # Extract mean according to method
            self.meanReturns = metrics.getMeanReturns(method=meanMethod)

            # Build covariance depending on shrinkage option
            # If shrinkage == 'ledoit' and sklearn is available, use LedoitWolf estimator on returns
            covVals = None
            if shrinkage == 'ledoit' and LedoitWolf is not None and getattr(metrics, 'data', None) is not None:
                try:
                    returns = metrics.data.pct_change().dropna()
                    lw = LedoitWolf().fit(returns)
                    # LedoitWolf outputs covariance of the input scale (daily returns) -> annualize it
                    covVals = lw.covariance_ * metrics.annualizationFactor
                except Exception:
                    covVals = None

            # If LedoitWolf not used or failed, fall back to sample covariance and optional numeric shrinkage
            if covVals is None:
                sampleCov = metrics.covMatrix.values if hasattr(metrics.covMatrix, 'values') else np.array(metrics.covMatrix)
                if isinstance(shrinkage, (float, int)) and 0.0 < float(shrinkage) <= 1.0:
                    avgVar = np.mean(np.diag(sampleCov))
                    target = np.eye(sampleCov.shape[0]) * avgVar
                    covVals = (1.0 - float(shrinkage)) * sampleCov + float(shrinkage) * target
                else:
                    covVals = sampleCov

            # Regularize and set
            covVals = covVals + np.eye(covVals.shape[0]) * regularization
            if hasattr(metrics.covMatrix, 'index'):
                self.covMatrix = type(metrics.covMatrix)(covVals, index=metrics.covMatrix.index, columns=metrics.covMatrix.columns)
            else:
                self.covMatrix = covVals
        else:
            self.meanReturns = meanReturns
            self.covMatrix = covMatrix

    def negativeSharpeRatio(self, weights, riskFreeRate = 0.0):
        """
        Objective function to be minimized for the Sharpe Ratio.
        """
        mu = self.meanReturns.values if hasattr(self.meanReturns, 'values') else np.array(self.meanReturns)
        cov = self.covMatrix.values if hasattr(self.covMatrix, 'values') else np.array(self.covMatrix)

        pReturn = float(np.dot(mu, weights))
        pStdDev = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))

        eps = 1e-12
        if pStdDev < eps:
            return 1e10

        sharpeRatio = (pReturn - riskFreeRate) / pStdDev
        return -float(sharpeRatio)

    def optimizePortfolio(self, riskFreeRate = 0.0):
        """
        Finds the optimal portfolio with the maximum Sharpe Ratio.
        """
        mu = self.meanReturns.values if hasattr(self.meanReturns, 'values') else np.array(self.meanReturns)
        numAssets = len(mu)
        initWeights = np.array([1/numAssets] * numAssets)   # start with equal weights

        bounds = tuple((0.0, 1.0) for _ in range(numAssets))   # no short selling, weights between 0 and 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},)  # weights must sum to 1

        optimalWeightsResult = minimize(
            self.negativeSharpeRatio,
            initWeights,
            args=(riskFreeRate,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints)

        if optimalWeightsResult.success:
            optimalWeights = optimalWeightsResult.x

            cov = self.covMatrix.values if hasattr(self.covMatrix, 'values') else np.array(self.covMatrix)
            optimalReturn = float(np.dot(mu, optimalWeights))
            optimalStdDev = float(np.sqrt(np.dot(optimalWeights.T, np.dot(cov, optimalWeights))))
            sharpeRatio = float((optimalReturn - riskFreeRate) / (optimalStdDev if optimalStdDev > 0 else 1e-12))

            return {
                'weights': optimalWeights,
                'return': optimalReturn,
                'volatility': optimalStdDev,
                'sharpeRatio': sharpeRatio}
        else:
            print("Optimization failed:", optimalWeightsResult.message)
            return None
