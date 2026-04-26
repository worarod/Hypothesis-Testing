"""
test_07_two_sample_variance.py
==============================
Two-Sample Variance Test  (F-test)
  H0: sigma1^2 = sigma2^2

Run this test first when sigma is unknown and n < 30,
to decide whether to use equal_var=True (pooled) or
equal_var=False (Welch) in test_02_two_sample_mean.py.
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


def two_sample_variance(
    *,
    data1: Optional[pd.Series] = None,
    data2: Optional[pd.Series] = None,
    s1: Optional[float] = None,
    n1: Optional[int] = None,
    s2: Optional[float] = None,
    n2: Optional[int] = None,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    Test H0: sigma1^2 = sigma2^2   (F = s1^2 / s2^2)

    Tip: For a one-sided 'greater' test, assign the group
         expected to have larger variance as group 1.

    Parameters
    ----------
    data1/data2 : raw observations for each group
    s1/n1       : sample std and size for group 1
    s2/n2       : sample std and size for group 2
    alternative : 'two-sided' | 'greater' | 'less'
    alpha       : significance level (default 0.05)
    """
    if data1 is not None:
        data1 = pd.to_numeric(data1.dropna(), errors="coerce").dropna()
        n1, s1 = len(data1), float(data1.std(ddof=1))
    if data2 is not None:
        data2 = pd.to_numeric(data2.dropna(), errors="coerce").dropna()
        n2, s2 = len(data2), float(data2.std(ddof=1))

    if any(v is None for v in [s1, n1, s2, n2]):
        raise ValueError("Provide data or (s, n) for both groups.")

    df1  = n1 - 1
    df2  = n2 - 1
    stat = s1**2 / s2**2
    sym  = alt_symbol(alternative)

    if alternative == "two-sided":
        cv_hi    = stats.f.ppf(1 - alpha / 2, df1, df2)
        cv_lo    = 1 / stats.f.ppf(1 - alpha / 2, df2, df1)
        cv_str   = f"{cv_lo:.4f}  /  {cv_hi:.4f}"
        rejected = stat < cv_lo or stat > cv_hi
        p        = 2 * min(stats.f.cdf(stat, df1, df2), stats.f.sf(stat, df1, df2))
    elif alternative == "greater":
        cv_hi    = stats.f.ppf(1 - alpha, df1, df2)
        cv_str   = f"{cv_hi:.4f}"
        rejected = stat > cv_hi
        p        = stats.f.sf(stat, df1, df2)
    else:
        cv_lo    = 1 / stats.f.ppf(1 - alpha, df2, df1)
        cv_str   = f"{cv_lo:.4f}"
        rejected = stat < cv_lo
        p        = stats.f.cdf(stat, df1, df2)

    next_step = (
        "Use equal_var=False (Welch) in two_sample_mean()"
        if rejected
        else "Use equal_var=True  (pooled) in two_sample_mean()"
    )

    extra = {
        "n1"             : n1,
        "s1"             : f"{s1:.4f}",
        "s1^2"           : f"{s1**2:.4f}",
        "n2"             : n2,
        "s2"             : f"{s2:.4f}",
        "s2^2"           : f"{s2**2:.4f}",
        "df1"            : df1,
        "df2"            : df2,
        "F = s1^2/s2^2"  : f"{stat:.4f}",
        "Next step"      : next_step,
    }

    return TestResult(
        test_name      = f"Two-Sample Variance Test (F, df1={df1}, df2={df2})",
        H0             = "sigma1^2 = sigma2^2",
        H1             = f"sigma1^2 {sym} sigma2^2",
        alpha          = alpha,
        test_statistic = round(stat, 4),
        critical_value = cv_str,
        p_value        = round(p, 4),
        decision       = decision_str(rejected),
        conclusion     = f"{'Reject' if rejected else 'Accept'} H0 at alpha={alpha}",
        extra          = extra,
    )


if __name__ == "__main__":

    print("\n-- Ex14: Men vs Women assembly time ----------------")
    r = two_sample_variance(s1=6.1, n1=11, s2=5.3, n2=14,
                            alternative="greater", alpha=0.01)
    print(r)

    print("\n-- Extra: two-sided --------------------------------")
    r = two_sample_variance(s1=2.39, n1=8, s2=2.98, n2=8,
                            alternative="two-sided", alpha=0.05)
    print(r)

    # --- Excel usage ---
    # loader = ExcelLoader("data.xlsx")
    # col1, col2 = loader.get_two_columns("time_men", "time_women")
    # r = two_sample_variance(data1=col1, data2=col2, alternative="greater")
    # print(r)
