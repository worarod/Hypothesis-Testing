"""
test_05_two_sample_proportion.py
=================================
ทดสอบผลต่างสัดส่วนของประชากร 2 กลุ่ม
  H₀: p₁ − p₂ = d₀

กรณี:
  d₀ = 0  → ใช้ pooled proportion p̂
  d₀ ≠ 0  → ใช้ unpooled SE

ตัวอย่างจากสไลด์:
  Ex12 — town vs county voters (greater, d0=0)
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
    ทดสอบ H₀: p₁ − p₂ = d₀

    Parameters
    ----------
    x1, n1 : จำนวนสำเร็จและขนาดตัวอย่างกลุ่ม 1
    x2, n2 : จำนวนสำเร็จและขนาดตัวอย่างกลุ่ม 2
    d0     : ค่าผลต่างที่ทดสอบ (default 0)
    """
    p1hat = x1 / n1
    p2hat = x2 / n2
    diff  = p1hat - p2hat
    sym   = alt_symbol(alternative)

    if d0 == 0:
        # pooled
        phat    = (x1 + x2) / (n1 + n2)
        qhat    = 1 - phat
        se      = np.sqrt(phat * qhat * (1 / n1 + 1 / n2))
        pool_info = {"p̂ (pooled)": f"{phat:.4f}", "q̂ (pooled)": f"{qhat:.4f}"}
    else:
        q1hat = 1 - p1hat
        q2hat = 1 - p2hat
        se    = np.sqrt(p1hat * q1hat / n1 + p2hat * q2hat / n2)
        pool_info = {}

    stat       = (diff - d0) / se
    cv, cv_str = z_critical(alpha, alternative)
    rejected   = reject_z(stat, cv, alternative)
    p          = p_value_z(stat, alternative)

    extra = {
        "n₁, x₁"    : f"{n1},  {x1}",
        "n₂, x₂"    : f"{n2},  {x2}",
        "p̂₁"        : f"{p1hat:.4f}",
        "p̂₂"        : f"{p2hat:.4f}",
        "p̂₁ − p̂₂"  : f"{diff:.4f}",
        **pool_info,
        "SE"         : f"{se:.4f}",
        "กรณี"       : f"{'pooled (d₀=0)' if d0==0 else 'unpooled (d₀≠0)'}",
    }

    return TestResult(
        test_name      = "Two-Sample Proportion Test (Z)",
        H0             = f"p₁ − p₂ = {d0}",
        H1             = f"p₁ − p₂ {sym} {d0}",
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

    print("\n── Example 12: Town vs County voters ───────────────")
    r = two_sample_proportion(
        x1=120, n1=200,
        x2=240, n2=500,
        d0=0, alternative="greater", alpha=0.025
    )
    print(r)

    print("\n── ตัวอย่างเพิ่ม: d0 ≠ 0 (unpooled) ───────────────")
    r = two_sample_proportion(
        x1=45, n1=100,
        x2=30, n2=100,
        d0=0.05, alternative="greater", alpha=0.05
    )
    print(r)
