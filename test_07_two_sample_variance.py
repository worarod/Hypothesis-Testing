"""
test_07_two_sample_variance.py
==============================
ทดสอบความเท่ากันของความแปรปรวนของประชากรปกติ 2 กลุ่ม
  H₀: σ₁² = σ₂²   (F-test)

⚠️  ควรรันก่อนเสมอเมื่อต้องการเลือก pooled vs Welch t-test
    ใน test_02_two_sample_mean.py (กรณี n<30, ไม่ทราบ σ)

ตัวอย่างจากสไลด์:
  Ex14 — men vs women assembly time (greater)
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
    # ── ข้อมูลดิบ ──
    data1: Optional[pd.Series] = None,
    data2: Optional[pd.Series] = None,
    # ── หรือค่าสถิติ ──
    s1: Optional[float] = None,
    n1: Optional[int] = None,
    s2: Optional[float] = None,
    n2: Optional[int] = None,
    # ── สมมติฐาน ──
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    ทดสอบ H₀: σ₁² = σ₂²

    หมายเหตุ: F = s₁²/s₂²
      ถ้าต้องการทดสอบ σ₁² > σ₂²  ควรให้กลุ่มที่คาดว่ามี variance มากกว่า = กลุ่ม 1

    Parameters
    ----------
    data1/data2 : pandas Series ข้อมูลดิบ
    s1/n1       : sample std และขนาดกลุ่ม 1
    s2/n2       : sample std และขนาดกลุ่ม 2
    """
    if data1 is not None:
        data1 = pd.to_numeric(data1.dropna(), errors="coerce").dropna()
        n1, s1 = len(data1), float(data1.std(ddof=1))
    if data2 is not None:
        data2 = pd.to_numeric(data2.dropna(), errors="coerce").dropna()
        n2, s2 = len(data2), float(data2.std(ddof=1))

    if any(v is None for v in [s1, n1, s2, n2]):
        raise ValueError("ต้องระบุ data หรือ s+n สำหรับทั้งสองกลุ่ม")

    df1  = n1 - 1
    df2  = n2 - 1
    stat = s1**2 / s2**2
    sym  = alt_symbol(alternative)

    # ── critical values ───────────────────────────────
    if alternative == "two-sided":
        cv_hi   = stats.f.ppf(1 - alpha / 2, df1, df2)
        cv_lo   = 1 / stats.f.ppf(1 - alpha / 2, df2, df1)
        cv_str  = f"{cv_lo:.4f}  /  {cv_hi:.4f}"
        rejected = stat < cv_lo or stat > cv_hi
        p        = 2 * min(stats.f.cdf(stat, df1, df2),
                           stats.f.sf(stat, df1, df2))
    elif alternative == "greater":
        cv_hi   = stats.f.ppf(1 - alpha, df1, df2)
        cv_str  = f"{cv_hi:.4f}"
        rejected = stat > cv_hi
        p        = stats.f.sf(stat, df1, df2)
    else:  # less
        cv_lo   = 1 / stats.f.ppf(1 - alpha, df2, df1)
        cv_str  = f"{cv_lo:.4f}"
        rejected = stat < cv_lo
        p        = stats.f.cdf(stat, df1, df2)

    extra = {
        "n₁"   : n1,
        "s₁"   : f"{s1:.4f}",
        "s₁²"  : f"{s1**2:.4f}",
        "n₂"   : n2,
        "s₂"   : f"{s2:.4f}",
        "s₂²"  : f"{s2**2:.4f}",
        "df₁"  : df1,
        "df₂"  : df2,
        "F = s₁²/s₂²": f"{stat:.4f}",
    }

    advice = ""
    if not rejected:
        advice = " → ใช้ equal_var=True (pooled) ใน two_sample_mean()"
    else:
        advice = " → ใช้ equal_var=False (Welch) ใน two_sample_mean()"

    return TestResult(
        test_name      = f"Two-Sample Variance Test (F, df₁={df1}, df₂={df2})",
        H0             = "σ₁² = σ₂²",
        H1             = f"σ₁² {sym} σ₂²",
        alpha          = alpha,
        test_statistic = round(stat, 4),
        critical_value = cv_str,
        p_value        = round(p, 4),
        decision       = decision_str(rejected),
        conclusion     = (
            f"{'ปฏิเสธ' if rejected else 'ยอมรับ'} H₀ ที่ α={alpha}{advice}"
        ),
        extra          = extra,
    )


# ─────────────────────────────────────────────
#  ตัวอย่างจากสไลด์
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n── Example 14: Men vs Women assembly time ───────────")
    r = two_sample_variance(s1=6.1, n1=11, s2=5.3, n2=14,
                            alternative="greater", alpha=0.01)
    print(r)

    print("\n── ตัวอย่าง: สองด้าน (two-sided) ────────────────────")
    r = two_sample_variance(s1=2.39, n1=8, s2=2.98, n2=8,
                            alternative="two-sided", alpha=0.05)
    print(r)

    # ── ใช้กับ Excel ──────────────────────────────────
    # loader = ExcelLoader("data.xlsx")
    # col1, col2 = loader.get_two_columns("time_men", "time_women")
    # r = two_sample_variance(data1=col1, data2=col2, alternative="greater")
    # print(r)
