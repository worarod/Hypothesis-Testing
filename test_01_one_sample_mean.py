"""
test_01_one_sample_mean.py
==========================
ทดสอบค่าเฉลี่ยของประชากร 1 กลุ่ม
  H₀: μ = μ₀

กรณีที่ครอบคลุม:
  Case 1 — ทราบค่า σ หรือ n ≥ 30          → Z-test
  Case 2 — ไม่ทราบ σ และ n < 30           → t-test (ประชากรแจกแจงปกติ)

ใช้กับ:
  • ข้อมูลดิบจาก Excel (pandas Series)
  • ค่าสถิติสรุป (xbar, n, sigma/s)

ตัวอย่างจากสไลด์:
  Ex2 — life span > 70 yrs   (Z, known σ)
  Ex3 — km driven > 20,000   (Z, large n)
  Ex4 — lubricant = 10 liters (t, small n)
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
    # ── ข้อมูลดิบ (เลือกอย่างใดอย่างหนึ่ง) ──
    data: Optional[pd.Series] = None,
    # ── หรือค่าสถิติสรุป ──
    xbar: Optional[float] = None,
    n: Optional[int] = None,
    sigma: Optional[float] = None,   # population std (ทราบ)
    s: Optional[float] = None,       # sample std (ไม่ทราบ population)
    # ── สมมติฐาน ──
    mu0: float,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    ทดสอบ H₀: μ = μ₀

    Parameters
    ----------
    data        : pandas Series ข้อมูลดิบ (แทน xbar/n/s ได้เลย)
    xbar        : ค่าเฉลี่ยตัวอย่าง
    n           : ขนาดตัวอย่าง
    sigma       : ส่วนเบี่ยงเบนมาตรฐานประชากร (ถ้าทราบ)
    s           : ส่วนเบี่ยงเบนมาตรฐานตัวอย่าง (ถ้าไม่ทราบ sigma)
    mu0         : ค่า μ ที่ต้องการทดสอบ
    alternative : "two-sided" | "greater" | "less"
    alpha       : ระดับนัยสำคัญ (default 0.05)

    Returns
    -------
    TestResult object (พิมพ์ได้เลย)
    """
    # ── คำนวณสถิติจากข้อมูลดิบ ──────────────────────
    if data is not None:
        data = pd.to_numeric(data.dropna(), errors="coerce").dropna()
        n    = len(data)
        xbar = float(data.mean())
        s    = float(data.std(ddof=1))

    if xbar is None or n is None:
        raise ValueError("ต้องระบุ data หรือ xbar+n")

    sym = alt_symbol(alternative)

    # ── เลือก Z หรือ t ───────────────────────────────
    use_z   = (sigma is not None) or (n >= 30)
    std_err = (sigma if sigma is not None else s) / np.sqrt(n)
    stat    = (xbar - mu0) / std_err

    if use_z:
        cv, cv_str  = z_critical(alpha, alternative)
        rejected    = reject_z(stat, cv, alternative)
        p           = p_value_z(stat, alternative)
        dist_name   = "Z-test"
        extra = {
            "n"          : n,
            "x̄"         : f"{xbar:.4f}",
            "σ (or s)"   : f"{(sigma or s):.4f}",
            "Std Error"  : f"{std_err:.4f}",
            "กรณี"       : f"{'ทราบ σ' if sigma else 'ไม่ทราบ σ แต่ n≥30'}",
        }
    else:
        df_val      = n - 1
        cv, cv_str  = t_critical(alpha, df_val, alternative)
        rejected    = reject_t(stat, cv, alternative)
        p           = p_value_t(stat, df_val, alternative)
        dist_name   = f"t-test (df={df_val})"
        extra = {
            "n"          : n,
            "x̄"         : f"{xbar:.4f}",
            "s"          : f"{s:.4f}",
            "df"         : df_val,
            "Std Error"  : f"{std_err:.4f}",
            "กรณี"       : "ไม่ทราบ σ และ n<30",
        }

    conclusion = (
        f"{'ปฏิเสธ' if rejected else 'ยอมรับ'} H₀: "
        f"μ {'≠' if rejected else '='} {mu0} ที่ α={alpha}"
    )

    return TestResult(
        test_name      = f"One-Sample Mean Test ({dist_name})",
        H0             = f"μ = {mu0}",
        H1             = f"μ {sym} {mu0}",
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

    print("\n── Example 2: Z-test, known σ, n=100 ──────────────")
    r = one_sample_mean(xbar=71.8, n=100, sigma=8.9, mu0=70,
                        alternative="greater", alpha=0.05)
    print(r)

    print("\n── Example 3: Z-test, unknown σ, n≥30 ─────────────")
    r = one_sample_mean(xbar=23500, n=100, s=3900, mu0=20000,
                        alternative="greater", alpha=0.05)
    print(r)

    print("\n── Example 4: t-test, unknown σ, n<30 ─────────────")
    raw = pd.Series([10.2, 9.7, 10.1, 10.3, 10.1, 9.8, 9.9, 10.4, 10.3, 9.8])
    r = one_sample_mean(data=raw, mu0=10.0, alternative="two-sided", alpha=0.01)
    print(r)

    # ── ใช้กับ Excel ──────────────────────────────────
    # loader = ExcelLoader("data.xlsx")
    # col = loader.get_column("score", sheet="Sheet1")
    # r = one_sample_mean(data=col, mu0=70, alternative="greater", alpha=0.05)
    # print(r)
