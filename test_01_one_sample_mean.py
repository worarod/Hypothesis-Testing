"""
test_01_one_sample_mean.py
==========================
One-Sample Mean Test
  H0: mu = mu0

Cases:
  Case 1 -- Known sigma  OR  n >= 30   ->  Z-test
  Case 2 -- Unknown sigma AND n < 30   ->  t-test (assumes normal population)

Input:
  (a) Raw data via pandas Series
  (b) Summary statistics: xbar, n, sigma / s
"""

import pandas as pd
import numpy as np
from typing import Optional
from utils import (
    Alternative, TestResult,
    z_critical, t_critical,
    reject_z, reject_t,
    p_value_z, p_value_t,
    decision_str, alt_symbol,
    ExcelLoader,
)


def one_sample_mean(
    *,
    data: Optional[pd.Series] = None,
    xbar: Optional[float] = None,
    n: Optional[int] = None,
    sigma: Optional[float] = None,
    s: Optional[float] = None,
    mu0: float,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    Test H0: mu = mu0

    Parameters
    ----------
    data        : raw observations as a pandas Series
    xbar        : sample mean
    n           : sample size
    sigma       : population std deviation (if known)
    s           : sample std deviation
    mu0         : hypothesized mean
    alternative : 'two-sided' | 'greater' | 'less'
    alpha       : significance level (default 0.05)
    """
    if data is not None:
        data = pd.to_numeric(data.dropna(), errors="coerce").dropna()
        n    = len(data)
        xbar = float(data.mean())
        s    = float(data.std(ddof=1))

    if xbar is None or n is None:
        raise ValueError("Provide either 'data' or both 'xbar' and 'n'.")

    sym     = alt_symbol(alternative)
    use_z   = (sigma is not None) or (n >= 30)
    std_err = (sigma if sigma is not None else s) / np.sqrt(n)
    stat    = (xbar - mu0) / std_err

    if use_z:
        cv, cv_str = z_critical(alpha, alternative)
        rejected   = reject_z(stat, cv, alternative)
        p          = p_value_z(stat, alternative)
        dist_name  = "Z-test"
        extra = {
            "n"         : n,
            "x-bar"     : f"{xbar:.4f}",
            "sigma/s"   : f"{(sigma or s):.4f}",
            "Std Error" : f"{std_err:.4f}",
            "Case"      : "Known sigma" if sigma else "Unknown sigma, n >= 30",
        }
    else:
        df_val     = n - 1
        cv, cv_str = t_critical(alpha, df_val, alternative)
        rejected   = reject_t(stat, cv, alternative)
        p          = p_value_t(stat, df_val, alternative)
        dist_name  = f"t-test (df={df_val})"
        extra = {
            "n"         : n,
            "x-bar"     : f"{xbar:.4f}",
            "s"         : f"{s:.4f}",
            "df"        : df_val,
            "Std Error" : f"{std_err:.4f}",
            "Case"      : "Unknown sigma, n < 30",
        }

    return TestResult(
        test_name      = f"One-Sample Mean Test ({dist_name})",
        H0             = f"mu = {mu0}",
        H1             = f"mu {sym} {mu0}",
        alpha          = alpha,
        test_statistic = round(stat, 4),
        critical_value = cv_str,
        p_value        = round(p, 4),
        decision       = decision_str(rejected),
        conclusion     = f"{'Reject' if rejected else 'Accept'} H0: mu {'!=' if rejected else '='} {mu0} at alpha={alpha}",
        extra          = extra,
    )


if __name__ == "__main__":

    print("\n-- Ex2: Z-test, known sigma, n=100 ----------------")
    r = one_sample_mean(xbar=71.8, n=100, sigma=8.9,
                        mu0=70, alternative="greater", alpha=0.05)
    print(r)

    print("\n-- Ex3: Z-test, unknown sigma, n >= 30 ------------")
    r = one_sample_mean(xbar=23500, n=100, s=3900,
                        mu0=20000, alternative="greater", alpha=0.05)
    print(r)

    print("\n-- Ex4: t-test, unknown sigma, n < 30 -------------")
    raw = pd.Series([10.2, 9.7, 10.1, 10.3, 10.1, 9.8, 9.9, 10.4, 10.3, 9.8])
    r = one_sample_mean(data=raw, mu0=10.0, alternative="two-sided", alpha=0.01)
    print(r)

    # --- Excel usage ---
    # loader = ExcelLoader("data.xlsx")
    # col = loader.get_column("score", sheet="Sheet1")
    # r = one_sample_mean(data=col, mu0=70, alternative="greater", alpha=0.05)
    # print(r)
