"""
run_all.py
==========
รันการทดสอบทุกเงื่อนไขพร้อมกันครั้งเดียว

Usage:
  python run_all.py
"""

def header(title):
    print(f"\n{'═'*56}")
    print(f"  {title}")
    print(f"{'═'*56}")


if __name__ == "__main__":

    print("\n" + "█"*56)
    print("  HYPOTHESIS TESTING — Chapter 7")
    print("  รันทุกไฟล์พร้อมกัน")
    print("█"*56)

    # ── 1. One-sample mean ────────────────────────────
    header("1. One-Sample Mean  (test_01)")
    from test_01_one_sample_mean import one_sample_mean
    import pandas as pd

    r = one_sample_mean(xbar=71.8, n=100, sigma=8.9,
                        mu0=70, alternative="greater", alpha=0.05)
    print(r)

    raw = pd.Series([10.2, 9.7, 10.1, 10.3, 10.1, 9.8, 9.9, 10.4, 10.3, 9.8])
    r = one_sample_mean(data=raw, mu0=10.0,
                        alternative="two-sided", alpha=0.01)
    print(r)

    # ── 2. Two-sample mean ────────────────────────────
    header("2. Two-Sample Mean  (test_02)")
    from test_02_two_sample_mean import two_sample_mean

    r = two_sample_mean(xbar1=81, n1=25, sigma1=5.2,
                        xbar2=76, n2=36, sigma2=3.4,
                        d0=0, alternative="two-sided", alpha=0.05)
    print(r)

    r = two_sample_mean(xbar1=92.255, n1=8, s1=2.39,
                        xbar2=92.733, n2=8, s2=2.98,
                        d0=0, alternative="two-sided",
                        alpha=0.05, equal_var=True)
    print(r)

    r = two_sample_mean(xbar1=24.2, n1=15, s1=10**0.5,
                        xbar2=23.9, n2=10, s2=20**0.5,
                        d0=0, alternative="two-sided",
                        alpha=0.10, equal_var=False)
    print(r)

    # ── 3. Paired t-test ──────────────────────────────
    header("3. Paired t-Test  (test_03)")
    from test_03_paired_mean import paired_mean

    before = pd.Series([195, 213, 247, 201, 187, 210, 215, 246, 294, 310])
    after  = pd.Series([187, 195, 221, 190, 175, 197, 199, 221, 278, 285])
    r = paired_mean(before=before, after=after,
                    d0=0, alternative="greater", alpha=0.05)
    print(r)

    # ── 4. One-sample proportion ──────────────────────
    header("4. One-Sample Proportion  (test_04)")
    from test_04_one_sample_proportion import one_sample_proportion

    r = one_sample_proportion(x=8, n=15, p0=0.7,
                              alternative="two-sided", alpha=0.10)
    print(r)

    r = one_sample_proportion(x=70, n=100, p0=0.6,
                              alternative="greater", alpha=0.05)
    print(r)

    # ── 5. Two-sample proportion ──────────────────────
    header("5. Two-Sample Proportion  (test_05)")
    from test_05_two_sample_proportion import two_sample_proportion

    r = two_sample_proportion(x1=120, n1=200, x2=240, n2=500,
                              d0=0, alternative="greater", alpha=0.025)
    print(r)

    # ── 6. One-sample variance ────────────────────────
    header("6. One-Sample Variance χ²  (test_06)")
    from test_06_one_sample_variance import one_sample_variance

    r = one_sample_variance(s=1.2, n=10, sigma0_sq=0.9**2,
                            alternative="greater", alpha=0.05)
    print(r)

    # ── 7. Two-sample variance ────────────────────────
    header("7. Two-Sample Variance F  (test_07)")
    from test_07_two_sample_variance import two_sample_variance

    r = two_sample_variance(s1=6.1, n1=11, s2=5.3, n2=14,
                            alternative="greater", alpha=0.01)
    print(r)

    print("\n✅  ครบทุกการทดสอบ")
