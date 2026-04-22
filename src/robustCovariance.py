#-------------------------------------------------------------------------------
# Name:        robustCovariance.py
# Purpose:     Alternative covariance estimators: EWMA, denoised (RMT),
#              and regime-aware.
#
# Author:      Sai Tarun Ganapavarapu
#
# Created:     03-31-2026
# Licence:     MIT License
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity


class RobustCovarianceEstimator:
    """
    Provides alternative covariance estimators that plug into the optimizer
    wherever the existing metrics.py covariance matrix is used.

    Parameters
    ----------
    method : str
        One of: 'ewma', 'denoised', 'regimeAware'
    **kwargs : dict
        Method-specific parameters:
        - ewma:          halflife (int, default=63)
        - denoised:      bandwidth (float, default=0.01)
        - regimeAware:  stressMultiplier (float, default=1.5)
    """

    VALID_METHODS = ('ewma', 'denoised', 'regimeAware')

    def __init__(self, method: str = 'ewma', **kwargs):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got '{method}'")
        self.method = method
        self.halflife = int(kwargs.get('halflife', 63))
        self.bandwidth = float(kwargs.get('bandwidth', 0.01))
        self.stressMultiplier = float(kwargs.get('stressMultiplier', 1.5))

    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estimate the covariance matrix.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series, shape (T, N).

        Returns
        -------
        np.ndarray
            Annualised covariance matrix of shape (N, N).
        """
        self._validate(returns)
        if self.method == 'ewma':
            return self._computeEwma(returns)
        elif self.method == 'denoised':
            return self._computeDenoised(returns)
        else:  # regimeAware
            covMatrix, _ = self.estimateWithRegime(returns)
            return covMatrix

    def estimateWithRegime(self, returns: pd.DataFrame) -> tuple[np.ndarray, str]:
        """
        Estimate covariance and detect the current market regime.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily return series.

        Returns
        -------
        tuple[np.ndarray, str]
            (covarianceMatrix, regimeLabel) where regimeLabel is
            'normal' or 'stressed'.
        """
        self._validate(returns)
        regime = self._detectRegime(returns)
        baseCov = self._computeEwma(returns)
        if regime == 'stressed':
            baseCov = baseCov * self.stressMultiplier
        return baseCov, regime

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _validate(self, returns: pd.DataFrame) -> None:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        if returns.empty:
            raise ValueError("returns DataFrame is empty")
        if returns.isnull().all(axis=None):
            raise ValueError("returns DataFrame contains only NaN values")

    # ---- EWMA ---------------------------------------------------------

    def _computeEwma(self, returns: pd.DataFrame) -> np.ndarray:
        """Exponentially weighted covariance matrix (annualised)."""
        lam = 1.0 - 2.0 / (self.halflife + 1)
        T = len(returns)
        powers = np.arange(T - 1, -1, -1, dtype=float)
        rawWeights = (1.0 - lam) * (lam ** powers)
        rawWeights /= rawWeights.sum()

        retVals = returns.values
        meanW = np.average(retVals, axis=0, weights=rawWeights)
        demeaned = retVals - meanW

        cov = (demeaned * rawWeights[:, None]).T @ demeaned
        cov = self._ensurePsd(cov)
        return cov * 252

    # ---- Denoised via Random Matrix Theory ----------------------------

    def _computeDenoised(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Marchenko-Pastur denoising (Lopez de Prado, 2020).
        """
        T, N = returns.shape
        q = T / N

        sampleCov = returns.cov().values * 252
        vols = np.sqrt(np.diag(sampleCov))
        vols = np.where(vols < 1e-12, 1e-12, vols)

        corr = sampleCov / np.outer(vols, vols)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        eigenvalues, eigenvectors = np.linalg.eigh(corr)

        lambdaPlus = self._marchenkoPasturLambdaPlus(eigenvalues, q)

        noiseMask = eigenvalues <= lambdaPlus
        if noiseMask.any():
            noiseMean = eigenvalues[noiseMask].mean()
            eigenvaluesDenoised = np.where(noiseMask, noiseMean, eigenvalues)
        else:
            eigenvaluesDenoised = eigenvalues.copy()

        corrDenoised = eigenvectors @ np.diag(eigenvaluesDenoised) @ eigenvectors.T

        diagSqrt = np.sqrt(np.diag(corrDenoised))
        diagSqrt = np.where(diagSqrt < 1e-12, 1e-12, diagSqrt)
        corrDenoised = corrDenoised / np.outer(diagSqrt, diagSqrt)
        np.fill_diagonal(corrDenoised, 1.0)

        covDenoised = corrDenoised * np.outer(vols, vols)
        return self._ensurePsd(covDenoised)

    def _marchenkoPasturLambdaPlus(self, eigenvalues: np.ndarray, q: float) -> float:
        """Estimate the Marchenko-Pastur upper bound Lambda+ via KDE fitting."""

        def mpPdf(x: np.ndarray, sigma2: float, q: float) -> np.ndarray:
            lamMinus = sigma2 * (1 - 1 / np.sqrt(q)) ** 2
            lamPlus = sigma2 * (1 + 1 / np.sqrt(q)) ** 2
            valid = (x >= lamMinus) & (x <= lamPlus)
            pdf = np.zeros_like(x, dtype=float)
            with np.errstate(invalid='ignore', divide='ignore'):
                pdf[valid] = (q / (2 * np.pi * sigma2 * x[valid])) * np.sqrt(
                    np.maximum((lamPlus - x[valid]) * (x[valid] - lamMinus), 0.0)
                )
            return pdf

        kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian').fit(
            eigenvalues.reshape(-1, 1)
        )
        xGrid = np.linspace(eigenvalues.min(), eigenvalues.max(), 200)

        def loss(params: list) -> float:
            sigma2 = params[0]
            if sigma2 <= 0:
                return 1e9
            theoretical = mpPdf(xGrid, sigma2, q)
            theoretical /= np.maximum(theoretical.sum() * (xGrid[1] - xGrid[0]), 1e-12)
            logEmpirical = kde.score_samples(xGrid.reshape(-1, 1))
            empirical = np.exp(logEmpirical)
            empirical /= np.maximum(empirical.sum() * (xGrid[1] - xGrid[0]), 1e-12)
            m = 0.5 * (theoretical + empirical + 1e-12)   # midpoint distribution for KL divergence
            kl = np.sum(empirical * np.log(empirical / m + 1e-12))
            return float(kl)

        result = minimize(loss, x0=[1.0], method='Nelder-Mead',
                          options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500})
        sigma2Fit = max(result.x[0], 1e-6)
        return sigma2Fit * (1 + 1 / np.sqrt(q)) ** 2

    # ---- Regime detection --------------------------------------------

    def _detectRegime(self, returns: pd.DataFrame) -> str:
        """
        Compare trailing 21-day realised vol to its 252-day rolling median.
        If current vol > 1.5 * median → 'stressed', else 'normal'.
        """
        portReturns = returns.mean(axis=1)
        rollingVol = portReturns.rolling(21).std() * np.sqrt(252)
        medianVol = rollingVol.rolling(252).median()

        currentVol = rollingVol.iloc[-1]
        currentMedian = medianVol.iloc[-1]

        if pd.isna(currentVol) or pd.isna(currentMedian) or currentMedian < 1e-12:
            return 'normal'

        return 'stressed' if currentVol > 1.5 * currentMedian else 'normal'

    # ---- Utility ------------------------------------------------------

    @staticmethod
    def _ensurePsd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Clip negative eigenvalues to epsilon to guarantee PSD."""
        matrix = (matrix + matrix.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, epsilon)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
