"""Data loading + turning raw features into yes/no "literals".  Owner: Yanin

Output format (see FORMATS.md):
  L : bool array (n_rows, n_literals)
  y : bool array (n_rows,)  True = positive (fraud / default)
  names : list[str] literal names, e.g. "LIMIT_BAL <= 50000"
"""
import numpy as np
import pandas as pd

# ---------- loaders: one per dataset (paths assume data/raw/) ----------

def load_default(path="data/raw/UCI_Credit_Card.csv"):
    """Default of Credit Card Clients (Kaggle CSV version of the UCI data)."""
    df = pd.read_csv(path).drop(columns=["ID"])
    y = df.pop("default.payment.next.month").astype(bool).values
    cat = ["SEX", "EDUCATION", "MARRIAGE"]
    return df, y, cat

def load_creditcard(path="data/raw/creditcard.csv"):
    """Credit Card Fraud (ULB). All numeric. 0.17% positives."""
    df = pd.read_csv(path)
    y = df.pop("Class").astype(bool).values
    return df, y, []

def load_givemesomecredit(path="data/raw/cs-training.csv"):
    """Give Me Some Credit. Has missing values (income, dependents)."""
    df = pd.read_csv(path, index_col=0)
    y = df.pop("SeriousDlqin2yrs").astype(bool).values
    return df, y, []

def load_bank(path="data/raw/bank-full.csv"):
    """Bank Marketing (UCI). Semicolon-separated, label 'y' = yes/no."""
    df = pd.read_csv(path, sep=";")
    y = (df.pop("y") == "yes").values
    cat = [c for c in df.columns if df[c].dtype == object]
    return df, y, cat

LOADERS = {"default": load_default, "creditcard": load_creditcard,
           "credit": load_givemesomecredit, "bank": load_bank}

# ---------- binarizer: fit on TRAIN only, then transform train and test ----------

class Binarizer:
    def __init__(self, n_bins=10, max_cat_values=12):
        self.n_bins = n_bins
        self.max_cat_values = max_cat_values
        self.specs = []          # list of (column, op, value)

    def fit(self, df, cat_cols=()):
        self.specs = []
        for col in df.columns:
            s = df[col]
            if s.isna().any():
                self.specs.append((col, "isna", None))
            s = s.dropna()
            if col in cat_cols or s.dtype == object or s.nunique() <= self.max_cat_values:
                for v in sorted(s.unique(), key=str):
                    self.specs.append((col, "==", v))
            else:
                qs = np.unique(np.quantile(s, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                for t in qs:
                    self.specs.append((col, "<=", t))
                    self.specs.append((col, ">", t))
        return self

    def transform(self, df):
        cols = []
        for col, op, v in self.specs:
            s = df[col]
            if op == "isna":
                cols.append(s.isna().values)
            elif op == "==":
                cols.append((s == v).values)
            elif op == "<=":
                cols.append((s <= v).fillna(False).values)
            else:
                cols.append((s > v).fillna(False).values)
        return np.column_stack(cols).astype(bool)

    @property
    def names(self):
        out = []
        for col, op, v in self.specs:
            if op == "isna":
                out.append(f"{col} is missing")
            elif isinstance(v, (float, np.floating)):
                out.append(f"{col} {op} {v:.4g}")
            else:
                out.append(f"{col} {op} {v}")
        return out
