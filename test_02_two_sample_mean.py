"""
test_02_two_sample_mean.py
==========================
Two-Sample Mean Test (independent groups)
  H0: mu1 - mu2 = d0

Cases:
  Case 1 -- Known sigma1, sigma2  OR  n1, n2 >= 30  ->  Z-test
  Case 2 -- Unknown sigma, equal variance,   n < 30  ->  Pooled t-test
  Case 3 -- Unknown sigma, unequal variance, n < 30  ->  Welch t-test
"""

import numpy as np
import pandas as pd
from typing import Optional
from utils import (
    Alternative, TestResult,
    z_critical, t_critical,
    reject_z, reject_t,
    p_value_z, p_value_t,
    decision_str, alt_symbol,
    ExcelLoader,
)


def two_sample_mean(
    *,
    data1: Optional[pd.Series] = None,
    data2: Optional[pd.Series] = None,
    xbar1: Optional[float] = None,
    xbar2: Optional[float] = None,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
    sigma1: Optional[float] = None,
    sigma2: Optional[float] = None,
    s1: Optional[float] = None,
    s2: Optional[float] = None,
    d0: float = 0,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
    equal_var: bool = True,
) -> TestResult:
    """
    Test H0: mu1 - mu2 = d0

    Parameters
    ----------
    data1/data2     : raw observations for each group
    xbar1/xbar2     : sample means
    n1/n2           : sample sizes
    sigma1/sigma2   : population std deviations (if known)
    s1/s2           : sample std deviations
    d0              : hypothesized difference (default 0)
    equal_var       : True -> pooled t,  False -> Welch t
    alternative     : 'two-sided' | 'greater' | 'less'
    alpha           : significance level (default 0.05)
    """
    if data1 is not None:
        data1 = pd.to_numeric(data1.dropna(), errors="coerce").dropna()
        n1, xbar1, s1 = len(data1), float(data1.mean()), float(data1.std(ddof=1))
    if data2 is not None:
        data2 = pd.to_numeric(data2.dropna(), errors="coerce").dropna()
        n2, xbar2, s2 = len(data2), float(data2.mean()), float(data2.std(ddof=1))

    if any(v is None for v in [xbar1, xbar2, n1, n2]):
        raise ValueError("Provide data or xbar+n for both groups.")

    sym  = alt_symbol(alternative)
    diff = xbar1 - xbar2

    known_sigma  = (sigma1 is not None) and (sigma2 is not None)
    large_sample = (n1 >= 30) and (n2 >= 30)

    if known_sigma or large_sample:
        s1_ = sigma1 or s1
        s2_ = sigma2 or s2
        se   = np.sqrt(s1_**2 / n1 + s2_**2 / n2)
        stat = (diff - d0) / se
        cv, cv_str = z_critical(alpha, alternative)
        rejected   = reject_z(stat, cv, alternative)
        p          = p_value_z(stat, alternative)
        dist_name  = "Z-test"
        extra = {
            "n1, x-bar1"     : f"{n1},  {xbar1:.4f}",
            "n2, x-bar2"     : f"{n2},  {xbar2:.4f}",
            "sigma1/s1"      : f"{s1_:.4f}",
            "sigma2/s2"      : f"{s2_:.4f}",
            "x-bar1 - x-bar2": f"{diff:.4f}",
            "Std Error"      : f"{se:.4f}",
            "Case"           : "Known sigma" if known_sigma else "Unknown sigma, n >= 30",
        }
    elif equal_var:
        sp2        = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
        sp         = np.sqrt(sp2)
        se         = sp * np.sqrt(1 / n1 + 1 / n2)
        stat       = (diff - d0) / se
        df_val     = n1 + n2 - 2
        cv, cv_str = t_critical(alpha, df_val, alternative)
        rejected   = reject_t(stat, cv, alternative)
        p          = p_value_t(stat, df_val, alternative)
        dist_name  = f"Pooled t-test (df={df_val})"
        extra = {
            "n1, x-bar1"     : f"{n1},  {xbar1:.4f}",
            "n2, x-bar2"     : f"{n2},  {xbar2:.4f}",
            "s1"             : f"{s1:.4f}",
            "s2"             : f"{s2:.4f}",
            "Sp"             : f"{sp:.4f}",
            "df"             : df_val,
            "x-bar1 - x-bar2": f"{diff:.4f}",
            "Case"           : "Unknown sigma, equal variance (pooled)",
        }
    else:
        se         = np.sqrt(s1**2 / n1 + s2**2 / n2)
        stat       = (diff - d0) / se
        num        = (s1**2 / n1 + s2**2 / n2) ** 2
        den        = (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
        df_val     = int(round(num / den))
        cv, cv_str = t_critical(alpha, df_val, alternative)
        rejected   = reject_t(stat, cv, alternative)
        p          = p_value_t(stat, df_val, alternative)
        dist_name  = f"Welch t-test (df={df_val})"
        extra = {
            "n1, x-bar1"     : f"{n1},  {xbar1:.4f}",
            "n2, x-bar2"     : f"{n2},  {xbar2:.4f}",
            "s1"             : f"{s1:.4f}",
            "s2"             : f"{s2:.4f}",
            "df (Welch)"     : df_val,
            "x-bar1 - x-bar2": f"{diff:.4f}",
            "Case"           : "Unknown sigma, unequal variance (Welch)",
        }

    return TestResult(
        test_name      = f"Two-Sample Mean Test ({dist_name})",
        H0             = f"mu1 - mu2 = {d0}",
        H1             = f"mu1 - mu2 {sym} {d0}",
        alpha          = alpha,
        test_statistic = round(stat, 4),
        critical_value = cv_str,
        p_value        = round(p, 4),
        decision       = decision_str(rejected),
        conclusion     = f"{'Reject' if rejected else 'Accept'} H0: mu1-mu2 {'!=' if rejected else '='} {d0} at alpha={alpha}",
        extra          = extra,
    )


if __name__ == "__main__":

    print("\n-- Ex5: Z-test, known sigma -------------------------")
    r = two_sample_mean(xbar1=81, n1=25, sigma1=5.2,
                        xbar2=76, n2=36, sigma2=3.4,
                        d0=0, alternative="two-sided", alpha=0.05)
    print(r)

    print("\n-- Ex6: Z-test, large n, d0=12 ---------------------")
    r = two_sample_mean(xbar1=86.7, n1=50, s1=6.28,
                        xbar2=77.8, n2=50, s2=5.61,
                        d0=12, alternative="less", alpha=0.05)
    print(r)

    print("\n-- Ex7: Pooled t-test, equal variance ---------------")
    r = two_sample_mean(xbar1=92.255, n1=8, s1=2.39,
                        xbar2=92.733, n2=8, s2=2.98,
                        d0=0, alternative="two-sided", alpha=0.05, equal_var=True)
    print(r)

    print("\n-- Ex8: Welch t-test, unequal variance --------------")
    r = two_sample_mean(xbar1=24.2, n1=15, s1=10**0.5,
                        xbar2=23.9, n2=10, s2=20**0.5,
                        d0=0, alternative="two-sided", alpha=0.10, equal_var=False)
    print(r)

    # --- Excel usage ---
    # loader = ExcelLoader("data.xlsx")
    # col1, col2 = loader.get_two_columns("group_a", "group_b")
    # r = two_sample_mean(data1=col1, data2=col2, alternative="two-sided")
    # print(r)
