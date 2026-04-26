"""
test_06_one_sample_variance.py
==============================
One-Sample Variance Test  (Chi-square)
  H0: sigma^2 = sigma0^2

Assumes a normally distributed population.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional
from utils import (
    Alternative, TestResult,
    decision_str, alt_symbol,
    ExcelLoader,
)


def one_sample_variance(
    *,
    data: Optional[pd.Series] = None,
    s: Optional[float] = None,
    n: Optional[int] = None,
    sigma0_sq: float,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    Test H0: sigma^2 = sigma0^2

    Parameters
    ----------
    data        : raw observations as a pandas Series
    s           : sample standard deviation
    n           : sample size
    sigma0_sq   : hypothesized variance. Pass sigma0**2 if you know sigma0.
    alternative : 'two-sided' | 'greater' | 'less'
    alpha       : significance level (default 0.05)
    """
    if data is not None:
        data = pd.to_numeric(data.dropna(), errors="coerce").dropna()
        n    = len(data)
        s    = float(data.std(ddof=1))

    if s is None or n is None:
        raise ValueError("Provide either 'data' or both 's' and 'n'.")

    df_val = n - 1
    stat   = df_val * s**2 / sigma0_sq
    sym    = alt_symbol(alternative)

    if alternative == "two-sided":
        cv_lo    = stats.chi2.ppf(alpha / 2, df_val)
        cv_hi    = stats.chi2.ppf(1 - alpha / 2, df_val)
        cv_str   = f"{cv_lo:.4f}  /  {cv_hi:.4f}"
        rejected = stat < cv_lo or stat > cv_hi
        p        = 2 * min(stats.chi2.cdf(stat, df_val), stats.chi2.sf(stat, df_val))
    elif alternative == "greater":
        cv_hi    = stats.chi2.ppf(1 - alpha, df_val)
        cv_str   = f"{cv_hi:.4f}"
        rejected = stat > cv_hi
        p        = stats.chi2.sf(stat, df_val)
    else:
        cv_lo    = stats.chi2.ppf(alpha, df_val)
        cv_str   = f"{cv_lo:.4f}"
        rejected = stat < cv_lo
        p        = stats.chi2.cdf(stat, df_val)

    extra = {
        "n"        : n,
        "s"        : f"{s:.4f}",
        "s^2"      : f"{s**2:.4f}",
        "sigma0^2" : f"{sigma0_sq:.4f}",
        "sigma0"   : f"{sigma0_sq**0.5:.4f}",
        "df"       : df_val,
    }

    return TestResult(
        test_name      = f"One-Sample Variance Test (chi-sq, df={df_val})",
        H0             = f"sigma^2 = {sigma0_sq}",
        H1             = f"sigma^2 {sym} {sigma0_sq}",
        alpha          = alpha,
        test_statistic = round(stat, 4),
        critical_value = cv_str,
        p_value        = round(p, 4),
        decision       = decision_str(rejected),
        conclusion     = f"{'Reject' if rejected else 'Accept'} H0 at alpha={alpha}: {'Variance is significantly different' if rejected else 'No evidence of change in variance'}",
        extra          = extra,
    )


if __name__ == "__main__":

    print("\n-- Ex13: Battery std > 0.9 year? ------------------")
    r = one_sample_variance(s=1.2, n=10, sigma0_sq=0.9**2,
                            alternative="greater", alpha=0.05)
    print(r)

    print("\n-- Extra: two-sided with raw data ------------------")
    raw = pd.Series([10.2, 9.7, 10.1, 10.3, 10.1, 9.8, 9.9, 10.4, 10.3, 9.8])
    r = one_sample_variance(data=raw, sigma0_sq=0.04, alternative="two-sided", alpha=0.05)
    print(r)

    # --- Excel usage ---
    # loader = ExcelLoader("data.xlsx")
    # col = loader.get_column("measurement")
    # r = one_sample_variance(data=col, sigma0_sq=1.0, alternative="greater")
    # print(r)
