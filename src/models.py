"""Model definitions and hyperparameter search spaces.

Every encoder lives inside a Pipeline, so it is fitted on training rows only.
"""
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def onehot():
    """One-hot encode the category columns; pass numeric columns through unchanged."""
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"),
          make_column_selector(dtype_include="category"))],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


# --- Trivial baseline -------------------------------------------------------
def dummy():
    """Predicts the training positive rate (about 0.117) for every client."""
    return DummyClassifier(strategy="prior")


# --- Baseline: decision tree, depth chosen by 5-fold CV ---------------------
def tree(seed):
    return Pipeline([("prep", onehot()),
                     ("clf", DecisionTreeClassifier(random_state=seed))])


# The plan's grid stopped at min_samples_leaf=50, but CV picked 50 every time (the edge),
# so the grid was widened to 100/200/400 to find a real optimum.
TREE_GRID = {
    "clf__max_depth": list(range(2, 16)) + [None],
    "clf__min_samples_leaf": [1, 5, 20, 50, 100, 200, 400],
}


# --- Main model: random forest ---------------------------------------------
def forest(seed, **params):
    return Pipeline([("prep", onehot()),
                     ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                    random_state=seed, **params))])


FOREST_GRID = {
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "max_features": ["sqrt", 0.5],
}


# --- Improvement: class-weighted gradient boosting ---------------------------
def hgb(seed, weighted=True, **params):
    """HistGradientBoosting with native categorical support (reads the category dtype).

    early_stopping holds out 10% of the training rows internally to decide when to stop.
    """
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        class_weight="balanced" if weighted else None,
        max_iter=1000, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, scoring="loss",
        random_state=seed, **params)


HGB_GRID = {
    "learning_rate": [0.03, 0.1],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 50],
    "l2_regularization": [0.0, 1.0],
}
