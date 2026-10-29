"""Load bank-full.csv, build the two feature sets, and make the leak-free split."""
import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "bank-full.csv"

LEAK = "duration"          # only known after the call ends
TEST_SEED = 42             # the test set is always drawn with this seed
SEEDS = (42, 43, 44)       # seeds for the train/validation split and the models
FEATURE_SETS = ("DEPLOYABLE", "FULL")


def load(path=DATA_PATH):
    """Return features X (DataFrame) and target y (0/1 Series)."""
    df = pd.read_csv(path, sep=";")
    assert df.shape == (45211, 17), f"unexpected shape {df.shape}: is this bank-full.csv?"

    y = (df.pop("y") == "yes").astype(int).rename("y")

    # pdays = -1 means "never contacted before"; make that explicit as a flag.
    df["previously_contacted"] = (df["pdays"] != -1).astype(int)

    # Text columns -> pandas "category" dtype. This only records the list of labels
    # (e.g. job = admin., blue-collar, ...), no statistics, so it leaks nothing.
    # "unknown" is kept as a normal category on purpose.
    text_cols = df.select_dtypes(exclude="number").columns
    df[text_cols] = df[text_cols].astype("category")
    return df, y


def feature_set(X, name):
    """FULL keeps every column; DEPLOYABLE drops the leaky `duration`."""
    if name == "FULL":
        return X
    if name == "DEPLOYABLE":
        return X.drop(columns=[LEAK])
    raise ValueError(name)


def split(X, y, seed):
    """Stratified 70/15/15 split.

    The test set (15%) is fixed by TEST_SEED and is the same for every seed and model.
    `seed` only changes how the remaining 85% is divided into train (70%) and validation (15%).
    """
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=TEST_SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15 / 0.85, stratify=y_dev, random_state=seed)

    # Safety check: the three parts never share a row.
    assert not (set(X_tr.index) & set(X_val.index))
    assert not (set(X_dev.index) & set(X_test.index))
    return X_tr, X_val, X_test, y_tr, y_val, y_test
