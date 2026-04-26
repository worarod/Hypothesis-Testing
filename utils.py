"""
utils.py
========
Shared utilities for all hypothesis test modules.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Literal

Alternative = Literal["two-sided", "greater", "less"]


@dataclass
class TestResult:
    test_name: str
    H0: str
    H1: str
    alpha: float
    test_statistic: float
    critical_value: str
    p_value: float
    decision: str
    conclusion: str
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        sep = "-" * 56
        lines = [sep, f"  {self.test_name}", sep,
                 f"  H0 : {self.H0}", f"  H1 : {self.H1}",
                 f"  alpha = {self.alpha}", sep]
        for k, v in self.extra.items():
            lines.append(f"  {k:<24} = {v}")
        if self.extra:
            lines.append(sep)
        lines += [
            f"  {'Test Statistic':<24} = {self.test_statistic:.4f}",
            f"  Critical Value           : {self.critical_value}",
            f"  p-value                  = {self.p_value:.4f}",
            sep,
            f"  Decision   : {self.decision}",
            f"  Conclusion : {self.conclusion}",
            sep,
        ]
        return "\n".join(lines)


def z_critical(alpha, alternative):
    if alternative == "two-sided":
        cv = stats.norm.ppf(1 - alpha / 2)
        return cv, f"+/-{cv:.4f}"
    elif alternative == "greater":
        cv = stats.norm.ppf(1 - alpha)
        return cv, str(round(cv, 4))
    else:
        cv = stats.norm.ppf(alpha)
        return cv, str(round(cv, 4))


def t_critical(alpha, df, alternative):
    if alternative == "two-sided":
        cv = stats.t.ppf(1 - alpha / 2, df)
        return cv, f"+/-{cv:.4f}"
    elif alternative == "greater":
        cv = stats.t.ppf(1 - alpha, df)
        return cv, str(round(cv, 4))
    else:
        cv = stats.t.ppf(alpha, df)
        return cv, str(round(cv, 4))


def reject_z(z, cv, alternative):
    if alternative == "two-sided": return abs(z) > cv
    elif alternative == "greater": return z > cv
    else: return z < cv

def reject_t(t, cv, alternative):
    return reject_z(t, cv, alternative)

def p_value_z(z, alternative):
    if alternative == "two-sided": return 2 * stats.norm.sf(abs(z))
    elif alternative == "greater": return stats.norm.sf(z)
    else: return stats.norm.cdf(z)

def p_value_t(t, df, alternative):
    if alternative == "two-sided": return 2 * stats.t.sf(abs(t), df)
    elif alternative == "greater": return stats.t.sf(t, df)
    else: return stats.t.cdf(t, df)

def decision_str(rejected):
    return "Reject H0" if rejected else "Fail to Reject H0"

def alt_symbol(alternative):
    return {"two-sided": "!=", "greater": ">", "less": "<"}[alternative]


class ExcelLoader:
    """Load data from an Excel file."""

    def __init__(self, filepath):
        self.filepath = filepath
        self._cache = {}

    def sheet_names(self):
        return pd.ExcelFile(self.filepath).sheet_names

    def load_sheet(self, sheet=0):
        key = str(sheet)
        if key not in self._cache:
            df = pd.read_excel(self.filepath, sheet_name=sheet)
            df.columns = df.columns.str.strip()
            self._cache[key] = df
        return self._cache[key].copy()

    def get_column(self, column, sheet=0):
        df = self.load_sheet(sheet)
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found. Available: {list(df.columns)}")
        return pd.to_numeric(df[column], errors="coerce").dropna().reset_index(drop=True)

    def get_two_columns(self, col1, col2, sheet=0):
        df = self.load_sheet(sheet)
        sub = df[[col1, col2]].apply(pd.to_numeric, errors="coerce").dropna()
        return sub[col1].reset_index(drop=True), sub[col2].reset_index(drop=True)
