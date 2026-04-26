# Hypothesis Testing — Chapter 7

Python source code สำหรับงานวิจัย  
แยกเป็น **1 ไฟล์ต่อ 1 เงื่อนไข** ดึงข้อมูลจาก Excel ผ่าน pandas

> Chapter 7 · Assoc.Prof.Dr. Nitima Aschariyaphotha

---

## โครงสร้างไฟล์

```
hypothesis-testing/
├── utils.py                          ← helper กลาง (TestResult, critical values)
│
├── test_01_one_sample_mean.py        ← ค่าเฉลี่ย 1 กลุ่ม       (Z / t)
├── test_02_two_sample_mean.py        ← ค่าเฉลี่ย 2 กลุ่ม       (Z / t-pooled / t-Welch)
├── test_03_paired_mean.py            ← Paired t-test
├── test_04_one_sample_proportion.py  ← สัดส่วน 1 กลุ่ม         (Z)
├── test_05_two_sample_proportion.py  ← สัดส่วน 2 กลุ่ม         (Z)
├── test_06_one_sample_variance.py    ← ความแปรปรวน 1 กลุ่ม     (χ²)
├── test_07_two_sample_variance.py    ← ความแปรปรวน 2 กลุ่ม     (F)
│
├── run_all.py                        ← รันทุกไฟล์ครั้งเดียว
└── requirements.txt
```

---

## การทดสอบที่รองรับ

| ไฟล์ | การทดสอบ | สถิติ | กรณี |
|------|----------|-------|------|
| `test_01` | ค่าเฉลี่ย 1 กลุ่ม | Z | ทราบ σ หรือ n ≥ 30 |
| `test_01` | ค่าเฉลี่ย 1 กลุ่ม | t | ไม่ทราบ σ, n < 30 |
| `test_02` | ค่าเฉลี่ย 2 กลุ่ม | Z | ทราบ σ หรือ n ≥ 30 |
| `test_02` | ค่าเฉลี่ย 2 กลุ่ม | t pooled | ไม่ทราบ σ, equal variance |
| `test_02` | ค่าเฉลี่ย 2 กลุ่ม | t Welch | ไม่ทราบ σ, unequal variance |
| `test_03` | Paired t-test | t | ข้อมูลสัมพันธ์กัน |
| `test_04` | สัดส่วน 1 กลุ่ม | Z | — |
| `test_05` | สัดส่วน 2 กลุ่ม | Z | pooled / unpooled |
| `test_06` | ความแปรปรวน 1 กลุ่ม | χ² | — |
| `test_07` | ความแปรปรวน 2 กลุ่ม | F | — |

รองรับ `alternative = "two-sided"` / `"greater"` / `"less"` ทุกไฟล์

---

## ติดตั้ง

```bash
pip install -r requirements.txt
```

---

## การใช้งาน

### รัน demo ทั้งหมด
```bash
python run_all.py
```

### รันแยกทีละไฟล์
```bash
python test_01_one_sample_mean.py
python test_03_paired_mean.py
# ...
```

### ใช้ใน code งานวิจัยของคุณ

```python
import pandas as pd
from test_01_one_sample_mean import one_sample_mean
from test_03_paired_mean import paired_mean
from test_07_two_sample_variance import two_sample_variance
from utils import ExcelLoader

# ── โหลดจาก Excel ──────────────────────────────────────
loader = ExcelLoader("my_data.xlsx")

# ตัวอย่าง: one-sample mean จาก raw data
col = loader.get_column("score", sheet="Sheet1")
r = one_sample_mean(data=col, mu0=70, alternative="greater", alpha=0.05)
print(r)

# ตัวอย่าง: paired t-test
before, after = loader.get_two_columns("pre", "post", sheet="Paired")
r = paired_mean(before=before, after=after, alternative="greater")
print(r)

# ตัวอย่าง: ใส่ค่าสรุปเอง
r = one_sample_mean(xbar=75.2, n=40, s=8.5, mu0=70,
                    alternative="greater", alpha=0.05)
print(r)
```

---

## รูปแบบ Excel ที่รองรับ

**Raw data sheet** — 1 คอลัมน์ต่อ 1 กลุ่ม
| group_a | group_b |
|---------|---------|
| 75.2 | 68.4 |
| 80.1 | 72.3 |

**Paired sheet** — before/after
| before | after |
|--------|-------|
| 195 | 187 |
| 213 | 195 |

---

## ตัวอย่าง Output

```
────────────────────────────────────────────────────────
  One-Sample Mean Test (Z-test)
────────────────────────────────────────────────────────
  H₀ : μ = 70
  H₁ : μ > 70
  α  = 0.05
────────────────────────────────────────────────────────
  n                          = 100
  x̄                         = 71.8000
  σ (or s)                   = 8.9000
  Std Error                  = 0.8900
  กรณี                       = ทราบ σ
────────────────────────────────────────────────────────
  Test Statistic             = 2.0225
  Critical Value             : 1.6449
  p-value                    = 0.0216
────────────────────────────────────────────────────────
  Decision  : Reject H₀
  Conclusion: ปฏิเสธ H₀: μ ≠ 70 ที่ α=0.05
────────────────────────────────────────────────────────
```

---

## License
MIT
