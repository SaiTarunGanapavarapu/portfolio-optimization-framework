from __future__ import annotations

#-------------------------------------------------------------------------------
# Name:        advancedMetrics.py
# Purpose:     Advanced portfolio performance metrics: VaR, CVaR, Sortino,
#              Calmar, Omega, drawdown analysis, and factor decomposition.
#
# Author:      Sai Tarun Ganapavarapu
#
# Created:     03-31-2026
# Licence:     MIT License
#-------------------------------------------------------------------------------
import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value at Risk
# ---------------------------------------------------------------------------

def computeVar(
    returns: pd.Series,
    alpha: float = 0.05,
    method: str = 'historical'
) -> float:
    """
    Compute Value at Risk (VaR).

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    alpha : float
        Left-tail probability (default 0.05 → 95% VaR).
    method : str
        'historical', 'parametric', or 'cornish_fisher'.

    Returns
    -------
    float
        VaR as a positive number representing the loss.
    """
    r = returns.dropna()
    if r.empty:
        return np.nan

    if method == 'historical':
        return float(-np.percentile(r, alpha * 100))

    elif method == 'parametric':
        mu, sigma = r.mean(), r.std(ddof=1)
        z = stats.norm.ppf(alpha)
        return float(-(mu + z * sigma))

    elif method == 'cornish_fisher':
        mu, sigma = r.mean(), r.std(ddof=1)
        skew = float(stats.skew(r))
        kurt = float(stats.kurtosis(r))
        z = stats.norm.ppf(alpha)
        zCf = (z
               + (z**2 - 1) * skew / 6
               + (z**3 - 3 * z) * kurt / 24
               - (2 * z**3 - 5 * z) * skew**2 / 36)
        return float(-(mu + zCf * sigma))

    else:
        raise ValueError(f"Unknown VaR method: '{method}'. Use 'historical', 'parametric', or 'cornish_fisher'.")


# ---------------------------------------------------------------------------
# Conditional Value at Risk (Expected Shortfall)
# ---------------------------------------------------------------------------

def computeCvar(
    returns: pd.Series,
    alpha: float = 0.05,
    method: str = 'historical'
) -> float:
    """
    Compute Conditional Value at Risk (CVaR / Expected Shortfall).

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    alpha : float
        Left-tail probability (default 0.05 → 95% CVaR).
    method : str
        'historical' or 'parametric'.

    Returns
    -------
    float
        CVaR as a positive number representing the expected loss.
    """
    r = returns.dropna()
    if r.empty:
        return np.nan

    if method == 'historical':
        threshold = np.percentile(r, alpha * 100)
        tail = r[r <= threshold]
        return float(-tail.mean()) if not tail.empty else float(-threshold)

    elif method == 'parametric':
        mu, sigma = r.mean(), r.std(ddof=1)
        zAlpha = stats.norm.ppf(alpha)
        cvar = -(mu - sigma * stats.norm.pdf(zAlpha) / alpha)
        return float(cvar)

    else:
        raise ValueError(f"Unknown CVaR method: '{method}'. Use 'historical' or 'parametric'.")


# ---------------------------------------------------------------------------
# Sortino Ratio
# ---------------------------------------------------------------------------

def computeSortino(
    returns: pd.Series,
    riskFreeRate: float = 0.04,
    periods: int = 252
) -> float:
    """
    Compute the Sortino ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    riskFreeRate : float
        Annualised risk-free rate.
    periods : int
        Trading days per year for annualisation.

    Returns
    -------
    float
    """
    r = returns.dropna()
    if r.empty:
        return np.nan

    dailyRf = riskFreeRate / periods
    excess = r - dailyRf
    downside = excess[excess < 0]

    if len(downside) == 0:
        return np.inf

    downsideDev = np.sqrt((downside**2).mean()) * np.sqrt(periods)
    annualisedExcess = excess.mean() * periods

    return float(annualisedExcess / downsideDev) if downsideDev > 1e-12 else np.inf


# ---------------------------------------------------------------------------
# Calmar Ratio
# ---------------------------------------------------------------------------

def computeCalmar(returns: pd.Series, periods: int = 252) -> float:
    """
    Compute the Calmar ratio (annualised return / max drawdown).

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    periods : int
        Trading days per year.

    Returns
    -------
    float
    """
    r = returns.dropna()
    if r.empty:
        return np.nan

    annualisedReturn = (1 + r.mean()) ** periods - 1

    cumReturns = (1 + r).cumprod()
    peak = cumReturns.expanding().max()
    drawdown = (cumReturns - peak) / peak
    maxDd = float(drawdown.min())

    if abs(maxDd) < 1e-12:
        return np.inf

    return float(annualisedReturn / abs(maxDd))


# ---------------------------------------------------------------------------
# Omega Ratio
# ---------------------------------------------------------------------------

def computeOmega(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Compute the Omega ratio.

    Omega = E[max(r - threshold, 0)] / E[max(threshold - r, 0)]

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    threshold : float
        Daily return threshold (default 0.0).

    Returns
    -------
    float
    """
    r = returns.dropna()
    if r.empty:
        return np.nan

    gains = np.maximum(r - threshold, 0.0).mean()
    losses = np.maximum(threshold - r, 0.0).mean()

    if losses < 1e-12:
        return np.inf

    return float(gains / losses)


# ---------------------------------------------------------------------------
# Drawdown Analysis
# ---------------------------------------------------------------------------

def computeDrawdownSeries(portfolioValues: pd.Series) -> pd.DataFrame:
    """
    Compute running drawdown and duration from portfolio value series.

    Parameters
    ----------
    portfolioValues : pd.Series
        Portfolio NAV series.

    Returns
    -------
    pd.DataFrame
        Columns: 'drawdown' (negative values), 'duration' (days since peak).
        Scalar summary stats stored in .attrs:
        'maxDrawdown', 'avgDrawdown', 'maxDuration'.
    """
    pv = portfolioValues.dropna()
    if pv.empty:
        return pd.DataFrame(columns=['drawdown', 'duration'])

    peak = pv.expanding().max()
    drawdown = (pv - peak) / peak

    duration = pd.Series(0, index=pv.index, dtype=int)
    counter = 0
    for i, dd in enumerate(drawdown):
        counter = counter + 1 if dd < 0 else 0
        duration.iloc[i] = counter

    df = pd.DataFrame({'drawdown': drawdown, 'duration': duration})

    ddOnly = drawdown[drawdown < 0]
    df.attrs['maxDrawdown'] = float(drawdown.min())
    df.attrs['avgDrawdown'] = float(ddOnly.mean()) if not ddOnly.empty else 0.0
    df.attrs['maxDuration'] = int(duration.max())

    return df


# ---------------------------------------------------------------------------
# Factor Decomposition
# ---------------------------------------------------------------------------

def computeFactorDecomposition(
    returns: pd.Series,
    marketReturns: pd.Series,
    riskFreeRate: float = 0.04,
    periods: int = 252
) -> dict:
    """
    Single-factor CAPM decomposition.

    Parameters
    ----------
    returns : pd.Series
        Portfolio daily return series.
    marketReturns : pd.Series
        Benchmark/market daily return series.
    riskFreeRate : float
        Annualised risk-free rate.
    periods : int
        Trading days per year.

    Returns
    -------
    dict
        Keys: 'alpha', 'beta', 'rSquared', 'informationRatio', 'trackingError'.
    """
    dailyRf = riskFreeRate / periods
    aligned = pd.concat(
        [returns.rename('port'), marketReturns.rename('mkt')], axis=1
    ).dropna()

    if len(aligned) < 2:
        return {'alpha': np.nan, 'beta': np.nan, 'rSquared': np.nan,
                'informationRatio': np.nan, 'trackingError': np.nan}

    portExcess = aligned['port'] - dailyRf
    mktExcess = aligned['mkt'] - dailyRf

    slope, intercept, rValue, _, _ = stats.linregress(mktExcess, portExcess)
    beta = float(slope)
    alphaDaily = float(intercept)
    alphaAnnualised = float((1 + alphaDaily) ** periods - 1)
    rSquared = float(rValue ** 2)

    activeReturns = aligned['port'] - aligned['mkt']
    trackingError = float(activeReturns.std(ddof=1) * np.sqrt(periods))
    alphaFromActive = float(activeReturns.mean() * periods)
    informationRatio = float(alphaFromActive / trackingError) if trackingError > 1e-12 else np.nan

    return {
        'alpha': alphaAnnualised,
        'beta': beta,
        'rSquared': rSquared,
        'informationRatio': informationRatio,
        'trackingError': trackingError,
    }


# ---------------------------------------------------------------------------
# Full Tearsheet
# ---------------------------------------------------------------------------

def fullTearsheet(
    portfolioValues: pd.Series,
    benchmarkValues: pd.Series | None = None,
    riskFreeRate: float = 0.04,
    periods: int = 252
) -> dict:
    """
    Aggregate all metrics into a single dictionary.

    Parameters
    ----------
    portfolioValues : pd.Series
        Portfolio NAV series.
    benchmarkValues : pd.Series, optional
        Benchmark NAV series. If provided, adds CAPM decomposition metrics.
    riskFreeRate : float
        Annualised risk-free rate.
    periods : int
        Trading days per year.

    Returns
    -------
    dict
        Keys: 'totalReturn', 'cagr', 'volatility', 'sharpe', 'sortino',
        'calmar', 'omega', 'maxDrawdown', 'avgDrawdown', 'maxDrawdownDuration',
        'var95', 'cvar95', 'var99', 'cvar99', 'skewness', 'kurtosis'.
        Additionally if benchmarkValues provided:
        'alpha', 'beta', 'rSquared', 'informationRatio', 'trackingError'.
    """
    pv = portfolioValues.dropna()
    returns = pv.pct_change().dropna()

    if pv.empty or returns.empty:
        return {}

    totalReturn = float(pv.iloc[-1] / pv.iloc[0] - 1)
    nDays = len(pv)
    cagr = float((pv.iloc[-1] / pv.iloc[0]) ** (periods / nDays) - 1)
    volatility = float(returns.std(ddof=1) * np.sqrt(periods))

    dailyRf = riskFreeRate / periods
    sharpe = float(
        (returns.mean() - dailyRf) * periods / (volatility if volatility > 1e-12 else np.nan)
    )

    sortino = computeSortino(returns, riskFreeRate=riskFreeRate, periods=periods)
    calmar = computeCalmar(returns, periods=periods)
    omega = computeOmega(returns, threshold=0.0)

    ddDf = computeDrawdownSeries(pv)
    maxDrawdown = ddDf.attrs.get('maxDrawdown', np.nan)
    avgDrawdown = ddDf.attrs.get('avgDrawdown', np.nan)
    maxDuration = ddDf.attrs.get('maxDuration', np.nan)

    var95 = computeVar(returns, alpha=0.05)
    cvar95 = computeCvar(returns, alpha=0.05)
    var99 = computeVar(returns, alpha=0.01)
    cvar99 = computeCvar(returns, alpha=0.01)

    skewness = float(stats.skew(returns.dropna()))
    kurtosis = float(stats.kurtosis(returns.dropna()))

    result = {
        'totalReturn': totalReturn,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'omega': omega,
        'maxDrawdown': maxDrawdown,
        'avgDrawdown': avgDrawdown,
        'maxDrawdownDuration': maxDuration,
        'var95': var95,
        'cvar95': cvar95,
        'var99': var99,
        'cvar99': cvar99,
        'skewness': skewness,
        'kurtosis': kurtosis,
    }

    if benchmarkValues is not None:
        benchReturns = benchmarkValues.pct_change().dropna()
        factorStats = computeFactorDecomposition(
            returns, benchReturns, riskFreeRate=riskFreeRate, periods=periods
        )
        result.update(factorStats)

    return result
