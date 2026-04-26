"""
test_06_one_sample_variance.py
==============================
ทดสอบความแปรปรวนของประชากรปกติ 1 กลุ่ม
  H₀: σ² = σ₀²   (Chi-square test)

ตัวอย่างจากสไลด์:
  Ex13 — battery std > 0.9 year?
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
    # ── ข้อมูลดิบ ──
    data: Optional[pd.Series] = None,
    # ── หรือค่าสถิติ ──
    s: Optional[float] = None,
    n: Optional[int] = None,
    # ── สมมติฐาน ──
    sigma0_sq: float,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    ทดสอบ H₀: σ² = σ₀²

    Parameters
    ----------
    data       : pandas Series ข้อมูลดิบ
    s          : sample std deviation
    n          : ขนาดตัวอย่าง
    sigma0_sq  : ค่า σ₀²  (ถ้าทราบ σ₀ ให้ส่ง sigma0_sq = sigma0**2)
    """
    if data is not None:
        data = pd.to_numeric(data.dropna(), errors="coerce").dropna()
        n    = len(data)
        s    = float(data.std(ddof=1))

    if s is None or n is None:
        raise ValueError("ต้องระบุ data หรือ s+n")

    df_val = n - 1
    stat   = df_val * s**2 / sigma0_sq
    sym    = alt_symbol(alternative)

    # ── critical values and p-value ──────────────────
    if alternative == "two-sided":
        cv_lo   = stats.chi2.ppf(alpha / 2, df_val)
        cv_hi   = stats.chi2.ppf(1 - alpha / 2, df_val)
        cv_str  = f"{cv_lo:.4f}  /  {cv_hi:.4f}"
        rejected = stat < cv_lo or stat > cv_hi
        p        = 2 * min(stats.chi2.cdf(stat, df_val),
                           stats.chi2.sf(stat, df_val))
    elif alternative == "greater":
        cv_hi   = stats.chi2.ppf(1 - alpha, df_val)
        cv_str  = f"{cv_hi:.4f}"
        rejected = stat > cv_hi
        p        = stats.chi2.sf(stat, df_val)
    else:  # less
        cv_lo   = stats.chi2.ppf(alpha, df_val)
        cv_str  = f"{cv_lo:.4f}"
        rejected = stat < cv_lo
        p        = stats.chi2.cdf(stat, df_val)

    extra = {
        "n"      : n,
        "s"      : f"{s:.4f}",
        "s²"     : f"{s**2:.4f}",
        "σ₀²"    : f"{sigma0_sq:.4f}",
        "σ₀"     : f"{sigma0_sq**0.5:.4f}",
        "df"     : df_val,
    }

    return TestResult(
        test_name      = f"One-Sample Variance Test (χ², df={df_val})",
        H0             = f"σ² = {sigma0_sq}",
        H1             = f"σ² {sym} {sigma0_sq}",
        alpha          = alpha,
        test_statistic = round(stat, 4),
        critical_value = cv_str,
        p_value        = round(p, 4),
        decision       = decision_str(rejected),
        conclusion     = (
            f"{'ปฏิเสธ' if rejected else 'ยอมรับ'} H₀ ที่ α={alpha}: "
            f"{'σ² มีนัยสำคัญ' if rejected else 'ไม่มีหลักฐานว่า σ² เปลี่ยน'}"
        ),
        extra          = extra,
    )


# ─────────────────────────────────────────────
#  ตัวอย่างจากสไลด์
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n── Example 13: Battery std > 0.9 year? ─────────────")
    r = one_sample_variance(s=1.2, n=10, sigma0_sq=0.9**2,
                            alternative="greater", alpha=0.05)
    print(r)

    print("\n── ตัวอย่างเพิ่ม: two-sided ─────────────────────────")
    raw = pd.Series([10.2, 9.7, 10.1, 10.3, 10.1, 9.8, 9.9, 10.4, 10.3, 9.8])
    r = one_sample_variance(data=raw, sigma0_sq=0.04,
                            alternative="two-sided", alpha=0.05)
    print(r)

    # ── ใช้กับ Excel ──────────────────────────────────
    # loader = ExcelLoader("data.xlsx")
    # col = loader.get_column("measurement")
    # r = one_sample_variance(data=col, sigma0_sq=1.0, alternative="greater")
    # print(r)
