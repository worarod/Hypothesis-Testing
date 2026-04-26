"""
test_05_two_sample_proportion.py
=================================
Two-Sample Proportion Test
  H0: p1 - p2 = d0

Cases:
  d0 = 0  ->  pooled proportion estimate
  d0 != 0 ->  unpooled standard error
"""

import numpy as np
from utils import (
    Alternative, TestResult,
    z_critical, reject_z, p_value_z,
    decision_str, alt_symbol,
)


def two_sample_proportion(
    *,
    x1: int, n1: int,
    x2: int, n2: int,
    d0: float = 0,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    Test H0: p1 - p2 = d0

    Parameters
    ----------
    x1, n1      : successes and sample size for group 1
    x2, n2      : successes and sample size for group 2
    d0          : hypothesized difference (default 0)
    alternative : 'two-sided' | 'greater' | 'less'
    alpha       : significance level (default 0.05)
    """
    p1hat = x1 / n1
    p2hat = x2 / n2
    diff  = p1hat - p2hat
    sym   = alt_symbol(alternative)

    if d0 == 0:
        phat      = (x1 + x2) / (n1 + n2)
        qhat      = 1 - phat
        se        = np.sqrt(phat * qhat * (1 / n1 + 1 / n2))
        pool_info = {"p-hat (pooled)": f"{phat:.4f}", "q-hat (pooled)": f"{qhat:.4f}"}
    else:
        se        = np.sqrt(p1hat*(1-p1hat)/n1 + p2hat*(1-p2hat)/n2)
        pool_info = {}

    stat       = (diff - d0) / se
    cv, cv_str = z_critical(alpha, alternative)
    rejected   = reject_z(stat, cv, alternative)
    p          = p_value_z(stat, alternative)

    extra = {
        "n1, x1"          : f"{n1},  {x1}",
        "n2, x2"          : f"{n2},  {x2}",
        "p-hat1"          : f"{p1hat:.4f}",
        "p-hat2"          : f"{p2hat:.4f}",
        "p-hat1 - p-hat2" : f"{diff:.4f}",
        **pool_info,
        "Std Error"       : f"{se:.4f}",
        "Case"            : "pooled (d0=0)" if d0 == 0 else "unpooled (d0 != 0)",
    }

    return TestResult(
        test_name      = "Two-Sample Proportion Test (Z)",
        H0             = f"p1 - p2 = {d0}",
        H1             = f"p1 - p2 {sym} {d0}",
        alpha          = alpha,
        test_statistic = round(stat, 4),
        critical_value = cv_str,
        p_value        = round(p, 4),
        decision       = decision_str(rejected),
        conclusion     = f"{'Reject' if rejected else 'Accept'} H0 at alpha={alpha}",
        extra          = extra,
    )


if __name__ == "__main__":

    print("\n-- Ex12: Town vs County voters ---------------------")
    r = two_sample_proportion(x1=120, n1=200, x2=240, n2=500,
                              d0=0, alternative="greater", alpha=0.025)
    print(r)

    print("\n-- Extra: unpooled (d0 != 0) -----------------------")
    r = two_sample_proportion(x1=45, n1=100, x2=30, n2=100,
                              d0=0.05, alternative="greater", alpha=0.05)
    print(r)
