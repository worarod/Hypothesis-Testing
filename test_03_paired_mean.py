"""
test_03_paired_mean.py
======================
Paired t-Test
  H0: mu_D = d0

Use when: two related measurements on the same subject (before/after, repeated measures).
"""

import numpy as np
import pandas as pd
from typing import Optional
from utils import (
    Alternative, TestResult,
    t_critical, reject_t, p_value_t,
    decision_str, alt_symbol,
    ExcelLoader,
)


def paired_mean(
    *,
    before: Optional[pd.Series] = None,
    after:  Optional[pd.Series] = None,
    differences: Optional[pd.Series] = None,
    d0: float = 0,
    alternative: Alternative = "greater",
    alpha: float = 0.05,
) -> TestResult:
    """
    Test H0: mu_D = d0

    Parameters
    ----------
    before      : measurements before treatment
    after       : measurements after treatment
    differences : pre-computed D_i = before - after (alternative input)
    d0          : hypothesized mean difference (default 0)
    alternative : 'two-sided' | 'greater' | 'less'
    alpha       : significance level (default 0.05)
    """
    if differences is None:
        if before is None or after is None:
            raise ValueError("Provide 'before' and 'after', or 'differences'.")
        d_series = (
            pd.to_numeric(before, errors="coerce")
            - pd.to_numeric(after, errors="coerce")
        ).dropna()
    else:
        d_series = pd.to_numeric(differences.dropna(), errors="coerce").dropna()

    n    = len(d_series)
    dbar = float(d_series.mean())
    sd   = float(d_series.std(ddof=1))
    se   = sd / np.sqrt(n)
    stat = (dbar - d0) / se
    df   = n - 1

    sym        = alt_symbol(alternative)
    cv, cv_str = t_critical(alpha, df, alternative)
    rejected   = reject_t(stat, cv, alternative)
    p          = p_value_t(stat, df, alternative)

    extra = {
        "n (pairs)" : n,
        "D-bar"     : f"{dbar:.4f}",
        "S_D"       : f"{sd:.4f}",
        "Std Error" : f"{se:.4f}",
        "df"        : df,
    }

    if before is not None and after is not None:
        tbl = pd.DataFrame({
            "Before" : pd.to_numeric(before, errors="coerce").values,
            "After"  : pd.to_numeric(after,  errors="coerce").values,
            "D_i"    : d_series.values,
        })
        extra[""] = f"\n{tbl.to_string(index=False)}\n  Sum = {d_series.sum():.0f}"

    return TestResult(
        test_name      = f"Paired t-Test (df={df})",
        H0             = f"mu_D = {d0}",
        H1             = f"mu_D {sym} {d0}",
        alpha          = alpha,
        test_statistic = round(stat, 4),
        critical_value = cv_str,
        p_value        = round(p, 4),
        decision       = decision_str(rejected),
        conclusion     = f"{'Reject' if rejected else 'Accept'} H0 at alpha={alpha}: {'Significant difference found' if rejected else 'No sufficient evidence'}",
        extra          = extra,
    )


if __name__ == "__main__":

    print("\n-- Ex9: Diet program weight loss -------------------")
    before = pd.Series([195, 213, 247, 201, 187, 210, 215, 246, 294, 310])
    after  = pd.Series([187, 195, 221, 190, 175, 197, 199, 221, 278, 285])
    r = paired_mean(before=before, after=after, d0=0, alternative="greater", alpha=0.05)
    print(r)

    # --- Excel usage ---
    # loader = ExcelLoader("data.xlsx")
    # before, after = loader.get_two_columns("before", "after", sheet="Paired")
    # r = paired_mean(before=before, after=after, alternative="greater")
    # print(r)
