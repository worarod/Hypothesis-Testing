"""
test_04_one_sample_proportion.py
=================================
One-Sample Proportion Test
  H0: p = p0
"""

import numpy as np
import pandas as pd
from typing import Optional
from utils import (
    Alternative, TestResult,
    z_critical, reject_z, p_value_z,
    decision_str, alt_symbol,
    ExcelLoader,
)


def one_sample_proportion(
    *,
    data: Optional[pd.Series] = None,
    success_value=1,
    x: Optional[int] = None,
    n: Optional[int] = None,
    p0: float,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    Test H0: p = p0

    Parameters
    ----------
    data          : pandas Series of observations (0/1 or category)
    success_value : value counted as success (default 1)
    x             : number of successes
    n             : total sample size
    p0            : hypothesized proportion (0 < p0 < 1)
    alternative   : 'two-sided' | 'greater' | 'less'
    alpha         : significance level (default 0.05)
    """
    if data is not None:
        data = data.dropna()
        n    = len(data)
        x    = int((data == success_value).sum())

    if x is None or n is None:
        raise ValueError("Provide either 'data' or both 'x' and 'n'.")

    phat = x / n
    q0   = 1 - p0
    se   = np.sqrt(p0 * q0 / n)
    stat = (phat - p0) / se

    sym        = alt_symbol(alternative)
    cv, cv_str = z_critical(alpha, alternative)
    rejected   = reject_z(stat, cv, alternative)
    p          = p_value_z(stat, alternative)

    extra = {
        "n"              : n,
        "x (successes)"  : x,
        "p-hat = x/n"    : f"{phat:.4f}",
        "p0"             : f"{p0:.4f}",
        "q0 = 1 - p0"    : f"{q0:.4f}",
        "Std Error"      : f"{se:.4f}",
    }

    return TestResult(
        test_name      = "One-Sample Proportion Test (Z)",
        H0             = f"p = {p0}",
        H1             = f"p {sym} {p0}",
        alpha          = alpha,
        test_statistic = round(stat, 4),
        critical_value = cv_str,
        p_value        = round(p, 4),
        decision       = decision_str(rejected),
        conclusion     = f"{'Reject' if rejected else 'Accept'} H0 at alpha={alpha}",
        extra          = extra,
    )


if __name__ == "__main__":

    print("\n-- Ex10: Heat pumps 70%? (two-sided) ---------------")
    r = one_sample_proportion(x=8, n=15, p0=0.7, alternative="two-sided", alpha=0.10)
    print(r)

    print("\n-- Ex11: New drug > 60%? (greater) -----------------")
    r = one_sample_proportion(x=70, n=100, p0=0.6, alternative="greater", alpha=0.05)
    print(r)

    # --- Excel usage (0/1 column) ---
    # loader = ExcelLoader("data.xlsx")
    # col = loader.get_column("relief")
    # r = one_sample_proportion(data=col, p0=0.6, alternative="greater")
    # print(r)
