"""
test_04_one_sample_proportion.py
=================================
ทดสอบสัดส่วนของประชากร 1 กลุ่ม
  H₀: p = p₀

ตัวอย่างจากสไลด์:
  Ex10 — heat pumps 70%? (two-sided)
  Ex11 — new drug > 60% effective (greater)
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
    # ── ข้อมูลดิบ (0/1 หรือ category) ──
    data: Optional[pd.Series] = None,
    success_value = 1,
    # ── หรือจำนวนสำเร็จ ──
    x: Optional[int] = None,
    n: Optional[int] = None,
    # ── สมมติฐาน ──
    p0: float,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    """
    ทดสอบ H₀: p = p₀

    Parameters
    ----------
    data          : pandas Series (ค่า 0/1 หรือ category)
    success_value : ค่าที่ถือว่า "สำเร็จ" (default 1)
    x             : จำนวนสำเร็จ
    n             : ขนาดตัวอย่างทั้งหมด
    p0            : สัดส่วนที่ทดสอบ (0 < p0 < 1)
    """
    if data is not None:
        data = data.dropna()
        n    = len(data)
        x    = int((data == success_value).sum())

    if x is None or n is None:
        raise ValueError("ต้องระบุ data หรือ x+n")

    phat = x / n
    q0   = 1 - p0
    se   = np.sqrt(p0 * q0 / n)
    stat = (phat - p0) / se

    sym        = alt_symbol(alternative)
    cv, cv_str = z_critical(alpha, alternative)
    rejected   = reject_z(stat, cv, alternative)
    p          = p_value_z(stat, alternative)

    extra = {
        "n"            : n,
        "x (สำเร็จ)"  : x,
        "p̂ = x/n"     : f"{phat:.4f}",
        "p₀"           : f"{p0:.4f}",
        "q₀ = 1 − p₀" : f"{q0:.4f}",
        "SE"           : f"{se:.4f}",
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
        conclusion     = (
            f"{'ปฏิเสธ' if rejected else 'ยอมรับ'} H₀ ที่ α={alpha}"
        ),
        extra          = extra,
    )


# ─────────────────────────────────────────────
#  ตัวอย่างจากสไลด์
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n── Example 10: Heat pumps 70%? (two-sided) ─────────")
    r = one_sample_proportion(x=8, n=15, p0=0.7,
                              alternative="two-sided", alpha=0.10)
    print(r)

    print("\n── Example 11: New drug > 60%? (greater) ───────────")
    r = one_sample_proportion(x=70, n=100, p0=0.6,
                              alternative="greater", alpha=0.05)
    print(r)

    # ── ใช้กับ Excel (คอลัมน์ 0/1) ───────────────────
    # loader = ExcelLoader("data.xlsx")
    # col = loader.get_column("relief", sheet="Sheet1")  # 0 = ไม่หาย, 1 = หาย
    # r = one_sample_proportion(data=col, p0=0.6, alternative="greater")
    # print(r)
