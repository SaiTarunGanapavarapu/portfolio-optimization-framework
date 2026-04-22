from __future__ import annotations

#-------------------------------------------------------------------------------
# Name:        advancedOptimizer.py
# Purpose:     Advanced portfolio optimizers: CVaR minimisation, risk parity,
#              and maximum diversification.
#
# Author:      Sai Tarun Ganapavarapu
#
# Created:     03-31-2026
# Licence:     MIT License
#-------------------------------------------------------------------------------
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog

logger = logging.getLogger(__name__)


class AdvancedOptimizer:
    """
    Portfolio optimizer supporting three methods beyond classic mean-variance:
    - 'minCvar'            – Minimum Conditional Value-at-Risk (LP formulation)
    - 'riskParity'         – Equal risk contribution
    - 'maxDiversification' – Maximum diversification ratio

    Parameters
    ----------
    method : str
        One of 'minCvar', 'riskParity', 'maxDiversification'.
    riskFreeRate : float
        Annualised risk-free rate.
    maxWeight : float
        Maximum weight for any single asset (default 1.0 = unconstrained).
    minWeight : float
        Minimum weight for any single asset (default 0.0 = long-only).
    turnoverPenalty : float
        L1 turnover penalty coefficient added to the objective when
        prevWeights is provided (default 0.0 = no penalty).
    cvarAlpha : float
        Left-tail probability for CVaR (default 0.05 → 95% CVaR).
    """

    VALID_METHODS = ('minCvar', 'riskParity', 'maxDiversification')

    def __init__(
        self,
        method: str,
        riskFreeRate: float = 0.04,
        maxWeight: float = 1.0,
        minWeight: float = 0.0,
        turnoverPenalty: float = 0.0,
        cvarAlpha: float = 0.05,
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got '{method}'")
        self.method = method
        self.riskFreeRate = riskFreeRate
        self.maxWeight = maxWeight
        self.minWeight = minWeight
        self.turnoverPenalty = turnoverPenalty
        self.cvarAlpha = cvarAlpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        returns: pd.DataFrame,
        covMatrix: np.ndarray,
        meanReturns: np.ndarray,
        prevWeights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute optimal portfolio weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical daily return scenarios (T × N).
        covMatrix : np.ndarray
            Covariance matrix (N × N), annualised.
        meanReturns : np.ndarray
            Expected return vector (N,), annualised.
        prevWeights : np.ndarray, optional
            Previous weights for turnover penalty (N,).

        Returns
        -------
        np.ndarray
            Weight vector of shape (N,) summing to 1.
        """
        retVals = returns.values if isinstance(returns, pd.DataFrame) else np.asarray(returns)
        cov = np.asarray(covMatrix, dtype=float)
        mu = np.asarray(meanReturns, dtype=float)
        N = cov.shape[0]

        if retVals.shape[1] != N:
            raise ValueError("returns columns must match covMatrix dimension")

        if self.method == 'minCvar':
            return self._minCvar(retVals, N, prevWeights)
        elif self.method == 'riskParity':
            return self._riskParity(cov, N)
        else:  # maxDiversification
            return self._maxDiversification(cov, N, prevWeights)

    # ------------------------------------------------------------------
    # Minimum CVaR (Rockafellar & Uryasev 2000)
    # ------------------------------------------------------------------

    def _minCvar(
        self,
        scenarios: np.ndarray,
        N: int,
        prevWeights: np.ndarray | None,
    ) -> np.ndarray:
        """
        Minimise CVaR using the LP formulation.

        Decision variables: [w (N), z (1), u (T)]
        Minimise: z + 1/(T*alpha) * sum(u)
        Subject to:
          u_t >= -scenarios[t] @ w - z   for all t
          u_t >= 0
          sum(w) = 1
          minWeight <= w_i <= maxWeight
        """
        T = scenarios.shape[0]
        alpha = self.cvarAlpha
        useTurnover = self.turnoverPenalty > 0 and prevWeights is not None

        if not useTurnover:
            # Pure LP: variables [w_0..w_{N-1}, z, u_0..u_{T-1}]
            c = np.concatenate([
                np.zeros(N),
                [1.0],
                np.full(T, 1.0 / (T * alpha)),
            ])

            aUbRows = []
            for t in range(T):
                row = np.zeros(N + 1 + T)
                row[:N] = -scenarios[t]
                row[N] = -1.0
                row[N + 1 + t] = -1.0
                aUbRows.append(row)
            aUb = np.array(aUbRows)
            bUb = np.zeros(T)

            aEq = np.zeros((1, N + 1 + T))
            aEq[0, :N] = 1.0
            bEq = np.array([1.0])

            bounds = (
                [(self.minWeight, self.maxWeight)] * N
                + [(None, None)]
                + [(0.0, None)] * T
            )

            result = linprog(c, A_ub=aUb, b_ub=bUb, A_eq=aEq, b_eq=bEq,
                             bounds=bounds, method='highs')

            if result.success:
                return self._clipAndNormalise(result.x[:N])
            else:
                logger.warning("minCvar LP failed (%s); falling back to equal weights.", result.message)
                return np.full(N, 1.0 / N)

        else:
            prevW = np.asarray(prevWeights, dtype=float)

            def objective(w: np.ndarray) -> float:
                losses = -scenarios @ w
                zVal = np.percentile(losses, (1 - alpha) * 100)
                cvar = zVal + np.mean(np.maximum(losses - zVal, 0)) / alpha
                turnover = np.sum(np.abs(w - prevW))
                return float(cvar + self.turnoverPenalty * turnover)

            w0 = self._clipAndNormalise(prevW.copy())
            constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
            bounds = [(self.minWeight, self.maxWeight)] * N

            result = minimize(objective, w0, method='SLSQP', bounds=bounds,
                              constraints=constraints,
                              options={'ftol': 1e-9, 'maxiter': 1000})

            if result.success:
                return self._clipAndNormalise(result.x)
            else:
                logger.warning("minCvar SLSQP failed (%s); falling back to equal weights.", result.message)
                return np.full(N, 1.0 / N)

    # ------------------------------------------------------------------
    # Risk Parity
    # ------------------------------------------------------------------

    def _riskParity(self, cov: np.ndarray, N: int) -> np.ndarray:
        """Equal risk contribution (ERC) portfolio."""
        targetRc = 1.0 / N

        def objective(w: np.ndarray) -> float:
            sigma2 = float(w @ cov @ w)
            if sigma2 < 1e-20:
                return 1e10
            sigma = np.sqrt(sigma2)
            marginal = cov @ w
            rc = w * marginal / sigma
            rcNormalised = rc / sigma
            return float(np.sum((rcNormalised - targetRc) ** 2))

        w0 = np.full(N, 1.0 / N)
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
        bounds = [(max(self.minWeight, 1e-6), self.maxWeight)] * N

        result = minimize(objective, w0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'ftol': 1e-12, 'maxiter': 2000})

        if result.success or result.fun < 1e-6:
            return self._clipAndNormalise(result.x)
        else:
            logger.warning("riskParity did not converge (%s); returning equal weights.", result.message)
            return np.full(N, 1.0 / N)

    # ------------------------------------------------------------------
    # Maximum Diversification
    # ------------------------------------------------------------------

    def _maxDiversification(
        self, cov: np.ndarray, N: int, prevWeights: np.ndarray | None
    ) -> np.ndarray:
        """Maximise diversification ratio DR = (w^T σ) / sqrt(w^T Σ w)."""
        vols = np.sqrt(np.maximum(np.diag(cov), 1e-20))

        def objective(w: np.ndarray) -> float:
            portVar = float(w @ cov @ w)
            weightedVol = float(w @ vols)
            if abs(weightedVol) < 1e-12:
                return 1e10
            base = portVar / (weightedVol ** 2)
            if self.turnoverPenalty > 0 and prevWeights is not None:
                prevW = np.asarray(prevWeights, dtype=float)
                base += self.turnoverPenalty * np.sum(np.abs(w - prevW))
            return float(base)

        w0 = vols / vols.sum()
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
        bounds = [(self.minWeight, self.maxWeight)] * N

        result = minimize(objective, w0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'ftol': 1e-12, 'maxiter': 1000})

        if result.success:
            return self._clipAndNormalise(result.x)
        else:
            logger.warning("maxDiversification failed (%s); returning vol-weighted.", result.message)
            return vols / vols.sum()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _clipAndNormalise(self, weights: np.ndarray) -> np.ndarray:
        """Clip to [minWeight, maxWeight] and renormalise to sum=1."""
        w = np.clip(weights, self.minWeight, self.maxWeight)
        total = w.sum()
        if total < 1e-12:
            N = len(w)
            return np.full(N, 1.0 / N)
        return w / total
