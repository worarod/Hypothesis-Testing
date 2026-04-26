# Hypothesis Testing

A Python toolkit for statistical hypothesis testing.
Each test type is isolated in its own file for easy reuse in research workflows.
Data can be loaded directly from Excel via pandas.

---

## File Structure

```
hypothesis-testing/
├── utils.py                          ← shared helpers (TestResult, critical values, ExcelLoader)
│
├── test_01_one_sample_mean.py        ← one-sample mean test        (Z / t)
├── test_02_two_sample_mean.py        ← two-sample mean test        (Z / pooled-t / Welch-t)
├── test_03_paired_mean.py            ← paired t-test
├── test_04_one_sample_proportion.py  ← one-sample proportion test  (Z)
├── test_05_two_sample_proportion.py  ← two-sample proportion test  (Z)
├── test_06_one_sample_variance.py    ← one-sample variance test    (chi-square)
├── test_07_two_sample_variance.py    ← two-sample variance test    (F)
│
├── run_all.py                        ← run every test at once
└── requirements.txt
```

---

## Tests Covered

| File | Test | Statistic | Case |
|------|------|-----------|------|
| `test_01` | One-sample mean | Z | Known sigma or n >= 30 |
| `test_01` | One-sample mean | t | Unknown sigma, n < 30 |
| `test_02` | Two-sample mean | Z | Known sigma or n >= 30 |
| `test_02` | Two-sample mean | Pooled t | Unknown sigma, equal variance |
| `test_02` | Two-sample mean | Welch t | Unknown sigma, unequal variance |
| `test_03` | Paired t-test | t | Related samples (before/after) |
| `test_04` | One-sample proportion | Z | — |
| `test_05` | Two-sample proportion | Z | Pooled / unpooled |
| `test_06` | One-sample variance | Chi-square | — |
| `test_07` | Two-sample variance | F | — |

All tests support `alternative = "two-sided"` / `"greater"` / `"less"`.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Run all examples at once

```bash
python run_all.py
```

### Run a single test file

```bash
python test_01_one_sample_mean.py
python test_03_paired_mean.py
```

### Use in your own research code

```python
import pandas as pd
from test_01_one_sample_mean import one_sample_mean
from test_02_two_sample_mean import two_sample_mean
from test_03_paired_mean import paired_mean
from utils import ExcelLoader

# Load data from Excel
loader = ExcelLoader("my_data.xlsx")

# One-sample mean from raw data
col = loader.get_column("score", sheet="Sheet1")
r = one_sample_mean(data=col, mu0=70, alternative="greater", alpha=0.05)
print(r)

# Paired t-test from two columns
before, after = loader.get_two_columns("pre", "post", sheet="Paired")
r = paired_mean(before=before, after=after, alternative="greater", alpha=0.05)
print(r)

# Pass summary statistics directly (no Excel needed)
r = two_sample_mean(xbar1=81, n1=25, sigma1=5.2,
                    xbar2=76, n2=36, sigma2=3.4,
                    d0=0, alternative="two-sided", alpha=0.05)
print(r)
```

---

## Excel Format

**Raw data sheet** — one column per group

| group_a | group_b |
|---------|---------|
| 75.2 | 68.4 |
| 80.1 | 72.3 |

**Paired sheet** — before and after columns

| before | after |
|--------|-------|
| 195 | 187 |
| 213 | 195 |

---

## Sample Output

```
--------------------------------------------------------
  One-Sample Mean Test (Z-test)
--------------------------------------------------------
  H0 : mu = 70
  H1 : mu > 70
  alpha = 0.05
--------------------------------------------------------
  n                        = 100
  x-bar                    = 71.8000
  sigma/s                  = 8.9000
  Std Error                = 0.8900
  Case                     = Known sigma
--------------------------------------------------------
  Test Statistic           = 2.0225
  Critical Value           : 1.6449
  p-value                  = 0.0216
--------------------------------------------------------
  Decision   : Reject H0
  Conclusion : Reject H0: mu != 70 at alpha=0.05
--------------------------------------------------------
```

---

## License

MIT
