"""
test_02_two_sample_mean.py
==========================
ทดสอบผลต่างค่าเฉลี่ยของประชากร 2 กลุ่มอิสระ
  H₀: μ₁ − μ₂ = d₀

กรณีที่ครอบคลุม:
  Case 1 — ทราบ σ₁,σ₂ หรือ n₁,n₂ ≥ 30          → Z-test
  Case 2 — ไม่ทราบ σ แต่ σ₁²=σ₂², n<30          → t-test pooled
  Case 3 — ไม่ทราบ σ และ σ₁²≠σ₂², n<30          → t-test Welch

ตัวอย่างจากสไลด์:
  Ex5 — μ₁=μ₂? (Z, known σ)
  Ex6 — thread A − B < 12  (Z, large n)
  Ex7 — catalyst yield (t pooled, equal var)
  Ex8 — circuit design (t Welch, unequal var)
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
    # ── ข้อมูลดิบ ──
    data1: Optional[pd.Series] = None,
    data2: Optional[pd.Series] = None,
    # ── หรือค่าสถิติสรุป ──
    xbar1: Optional[float] = None,
    xbar2: Optional[float] = None,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
    sigma1: Optional[float] = None,
    sigma2: Optional[float] = None,
    s1: Optional[float] = None,
    s2: Optional[float] = None,
    # ── สมมติฐาน ──
    d0: float = 0,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
    equal_var: bool = True,
) -> TestResult:
    """
    ทดสอบ H₀: μ₁ − μ₂ = d₀

    Parameters
    ----------
    data1/data2   : pandas Series ข้อมูลดิบแต่ละกลุ่ม
    xbar1/xbar2   : ค่าเฉลี่ย
    n1/n2         : ขนาดตัวอย่าง
    sigma1/sigma2 : ส่วนเบี่ยงเบนประชากร (ถ้าทราบ)
    s1/s2         : ส่วนเบี่ยงเบนตัวอย่าง
    d0            : ค่าผลต่างที่ทดสอบ (default 0)
    equal_var     : True = pooled t,  False = Welch t
    """
    if data1 is not None:
        data1 = pd.to_numeric(data1.dropna(), errors="coerce").dropna()
        n1, xbar1, s1 = len(data1), float(data1.mean()), float(data1.std(ddof=1))
    if data2 is not None:
        data2 = pd.to_numeric(data2.dropna(), errors="coerce").dropna()
        n2, xbar2, s2 = len(data2), float(data2.mean()), float(data2.std(ddof=1))

    if any(v is None for v in [xbar1, xbar2, n1, n2]):
        raise ValueError("ต้องระบุ data หรือ xbar+n สำหรับทั้งสองกลุ่ม")

    sym  = alt_symbol(alternative)
    diff = xbar1 - xbar2

    known_sigma  = (sigma1 is not None) and (sigma2 is not None)
    large_sample = (n1 >= 30) and (n2 >= 30)

    # ── Case 1: Z-test ────────────────────────────────
    if known_sigma or large_sample:
        s1_ = sigma1 or s1
        s2_ = sigma2 or s2
        se  = np.sqrt(s1_**2 / n1 + s2_**2 / n2)
        stat = (diff - d0) / se
        cv, cv_str = z_critical(alpha, alternative)
        rejected   = reject_z(stat, cv, alternative)
        p          = p_value_z(stat, alternative)
        dist_name  = "Z-test"
        extra = {
            "n₁, x̄₁"        : f"{n1},  {xbar1:.4f}",
            "n₂, x̄₂"        : f"{n2},  {xbar2:.4f}",
            "σ₁ (or s₁)"    : f"{s1_:.4f}",
            "σ₂ (or s₂)"    : f"{s2_:.4f}",
            "x̄₁ − x̄₂"      : f"{diff:.4f}",
            "SE"             : f"{se:.4f}",
            "กรณี"           : f"{'ทราบ σ' if known_sigma else 'ไม่ทราบ σ แต่ n≥30'}",
        }

    # ── Case 2: t-test pooled ─────────────────────────
    elif equal_var:
        sp2       = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
        sp        = np.sqrt(sp2)
        se        = sp * np.sqrt(1 / n1 + 1 / n2)
        stat      = (diff - d0) / se
        df_val    = n1 + n2 - 2
        cv, cv_str = t_critical(alpha, df_val, alternative)
        rejected  = reject_t(stat, cv, alternative)
        p         = p_value_t(stat, df_val, alternative)
        dist_name = f"t-pooled (df={df_val})"
        extra = {
            "n₁, x̄₁"        : f"{n1},  {xbar1:.4f}",
            "n₂, x̄₂"        : f"{n2},  {xbar2:.4f}",
            "s₁"             : f"{s1:.4f}",
            "s₂"             : f"{s2:.4f}",
            "Sp (pooled std)": f"{sp:.4f}",
            "df"             : df_val,
            "x̄₁ − x̄₂"      : f"{diff:.4f}",
            "กรณี"           : "ไม่ทราบ σ, equal variance (pooled)",
        }

    # ── Case 3: t-test Welch ──────────────────────────
    else:
        se        = np.sqrt(s1**2 / n1 + s2**2 / n2)
        stat      = (diff - d0) / se
        num       = (s1**2 / n1 + s2**2 / n2) ** 2
        den       = (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
        df_val    = int(round(num / den))
        cv, cv_str = t_critical(alpha, df_val, alternative)
        rejected  = reject_t(stat, cv, alternative)
        p         = p_value_t(stat, df_val, alternative)
        dist_name = f"t-Welch (df={df_val})"
        extra = {
            "n₁, x̄₁"        : f"{n1},  {xbar1:.4f}",
            "n₂, x̄₂"        : f"{n2},  {xbar2:.4f}",
            "s₁"             : f"{s1:.4f}",
            "s₂"             : f"{s2:.4f}",
            "df (Welch)"     : df_val,
            "x̄₁ − x̄₂"      : f"{diff:.4f}",
            "กรณี"           : "ไม่ทราบ σ, unequal variance (Welch)",
        }

    conclusion = (
        f"{'ปฏิเสธ' if rejected else 'ยอมรับ'} H₀: "
        f"μ₁−μ₂ {'≠' if rejected else '='} {d0} ที่ α={alpha}"
    )

    return TestResult(
        test_name      = f"Two-Sample Mean Test ({dist_name})",
        H0             = f"μ₁ − μ₂ = {d0}",
        H1             = f"μ₁ − μ₂ {sym} {d0}",
        alpha          = alpha,
        test_statistic = round(stat, 4),
        critical_value = cv_str,
        p_value        = round(p, 4),
        decision       = decision_str(rejected),
        conclusion     = conclusion,
        extra          = extra,
    )


# ─────────────────────────────────────────────
#  ตัวอย่างจากสไลด์
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n── Example 5: Z-test, known σ ──────────────────────")
    r = two_sample_mean(xbar1=81, n1=25, sigma1=5.2,
                        xbar2=76, n2=36, sigma2=3.4,
                        d0=0, alternative="two-sided", alpha=0.05)
    print(r)

    print("\n── Example 6: Z-test, large n, d0=12 ───────────────")
    r = two_sample_mean(xbar1=86.7, n1=50, s1=6.28,
                        xbar2=77.8, n2=50, s2=5.61,
                        d0=12, alternative="less", alpha=0.05)
    print(r)

    print("\n── Example 7: t-test pooled, equal variance ─────────")
    r = two_sample_mean(xbar1=92.255, n1=8, s1=2.39,
                        xbar2=92.733, n2=8, s2=2.98,
                        d0=0, alternative="two-sided", alpha=0.05,
                        equal_var=True)
    print(r)

    print("\n── Example 8: t-test Welch, unequal variance ────────")
    r = two_sample_mean(xbar1=24.2, n1=15, s1=10**0.5,
                        xbar2=23.9, n2=10, s2=20**0.5,
                        d0=0, alternative="two-sided", alpha=0.10,
                        equal_var=False)
    print(r)

    # ── ใช้กับ Excel ──────────────────────────────────
    # loader = ExcelLoader("data.xlsx")
    # col1, col2 = loader.get_two_columns("group_a", "group_b")
    # r = two_sample_mean(data1=col1, data2=col2, alternative="two-sided")
    # print(r)
