#-------------------------------------------------------------------------------
# Name:        transactionCosts.py
# Purpose:     Transaction cost model: commission, spread, and market impact.
#
# Author:      Sai Tarun Ganapavarapu
#
# Created:     03-31-2026
# Licence:     MIT License
#-------------------------------------------------------------------------------

import numpy as np

class TransactionCostModel:
    """
    Models three sources of transaction cost:
    1. Commission - a flat rate on the notional traded.
    2. Half-spread - bid-ask spread cost on the notional traded.
    3. Market impact - Almgren-Chriss style square-root impact (only when ADV is provided).

    Parameters
    ----------
    commissionRate : float
        One-way commission as a fraction (default 10 bps = 0.001).
    spreadCost : float
        One-way half-spread as a fraction (default 5 bps = 0.0005).
    marketImpactCoeff : float
        Coefficient in the Almgren-Chriss square-root impact model (default 0.1).
    """

    def __init__(
        self,
        commissionRate: float = 0.001,
        spreadCost: float = 0.0005,
        marketImpactCoeff: float = 0.1,
    ):
        if commissionRate < 0:
            raise ValueError("commissionRate must be non-negative")
        if spreadCost < 0:
            raise ValueError("spreadCost must be non-negative")
        if marketImpactCoeff < 0:
            raise ValueError("marketImpactCoeff must be non-negative")

        self.commissionRate = commissionRate
        self.spreadCost = spreadCost
        self.marketImpactCoeff = marketImpactCoeff

    def computeCost(
        self,
        oldWeights: np.ndarray,
        newWeights: np.ndarray,
        portfolioValue: float,
        avgDailyVolume: np.ndarray | None = None,
    ) -> float:
        """
        Compute total transaction cost in dollars for a rebalance.

        Parameters
        ----------
        oldWeights : np.ndarray
            Current portfolio weights.
        newWeights : np.ndarray
            Target portfolio weights.
        portfolioValue : float
            Current portfolio NAV in dollars.
        avgDailyVolume : np.ndarray, optional
            Average daily dollar volume per asset. If provided, market impact
            is included in the cost calculation.

        Returns
        -------
        float
            Total transaction cost in dollars.

        Notes
        -----
        Cost breakdown:
        - Commission:     commissionRate  * Σ|Δw_i| * portfolioValue
        - Spread:         spreadCost      * Σ|Δw_i| * portfolioValue
        - Market impact:  Σ_i marketImpactCoeff
                            * sqrt(|Δw_i| * portfolioValue / ADV_i)
                            * |Δw_i| * portfolioValue
          (skipped if avgDailyVolume is None)
        """
        oldWeights = np.asarray(oldWeights, dtype=float)
        newWeights = np.asarray(newWeights, dtype=float)

        if oldWeights.shape != newWeights.shape:
            raise ValueError("oldWeights and newWeights must have the same shape")
        if portfolioValue < 0:
            raise ValueError("portfolioValue must be non-negative")

        deltaWeights = np.abs(newWeights - oldWeights)
        totalTurnover = deltaWeights.sum()
        notionalTraded = totalTurnover * portfolioValue

        commission = self.commissionRate * notionalTraded
        spread = self.spreadCost * notionalTraded

        marketImpact = 0.0
        if avgDailyVolume is not None:
            adv = np.asarray(avgDailyVolume, dtype=float)
            if adv.shape != deltaWeights.shape:
                raise ValueError("avgDailyVolume must have the same length as weights")
            for dw, advI in zip(deltaWeights, adv):
                notionalI = dw * portfolioValue
                if advI > 1e-12 and notionalI > 0:
                    marketImpact += (
                        self.marketImpactCoeff
                        * np.sqrt(notionalI / advI)
                        * notionalI
                    )

        totalCost = commission + spread + marketImpact

        return float(totalCost)
