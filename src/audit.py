"""
audit.py
========
Production-ready data-auditing utilities for pandas DataFrames.

All functions are pure (no side-effects, no printing) and return either a
pandas DataFrame or a plain Python scalar/dict so they compose easily in
pipelines, notebooks, or CLI tools.

Usage
-----
    import pandas as pd
    from audit import (
        dataframe_overview,
        missing_value_report,
        duplicate_report,
        datatype_summary,
        candidate_primary_keys,
        memory_usage_report,
        unique_value_counts,
    )

    df = pd.read_csv("data.csv")
    print(dataframe_overview(df))
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# 1. DataFrame Overview
# ---------------------------------------------------------------------------

def dataframe_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return a high-level structural summary of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to inspect.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame with the following columns:

        - ``rows``            : total number of rows
        - ``columns``         : total number of columns
        - ``total_cells``     : rows x columns
        - ``missing_cells``   : count of NaN / NaT / None values
        - ``missing_pct``     : missing_cells / total_cells as a percentage
        - ``duplicate_rows``  : number of fully duplicated rows
        - ``memory_mb``       : total deep memory usage in megabytes

    Examples
    --------
    >>> overview = dataframe_overview(df)
    >>> overview["rows"].iloc[0]
    1000
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__!r}.")

    rows, cols = df.shape
    total_cells = rows * cols
    missing_cells = int(df.isna().sum().sum())
    missing_pct = round(missing_cells / total_cells * 100, 4) if total_cells else 0.0
    duplicate_rows = int(df.duplicated().sum())
    memory_mb = round(df.memory_usage(deep=True).sum() / 1_048_576, 4)

    return pd.DataFrame(
        {
            "rows": [rows],
            "columns": [cols],
            "total_cells": [total_cells],
            "missing_cells": [missing_cells],
            "missing_pct": [missing_pct],
            "duplicate_rows": [duplicate_rows],
            "memory_mb": [memory_mb],
        }
    )


# ---------------------------------------------------------------------------
# 2. Missing Value Report
# ---------------------------------------------------------------------------

def missing_value_report(
    df: pd.DataFrame,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Return per-column missing-value statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to inspect.
    threshold : float, optional
        Only return columns whose missing percentage **exceeds** this value
        (0-100). Default ``0.0`` returns all columns with at least one
        missing value. Pass ``-1`` to include columns with zero missing
        values as well.

    Returns
    -------
    pd.DataFrame
        Indexed by column name with columns:

        - ``dtype``         : pandas dtype of the column
        - ``missing_count`` : number of missing values
        - ``missing_pct``   : percentage of missing values (0-100)
        - ``present_count`` : number of non-missing values
        - ``present_pct``   : percentage of non-missing values (0-100)

        Rows are sorted by ``missing_count`` descending.

    Raises
    ------
    TypeError
        If *df* is not a pandas DataFrame.
    ValueError
        If *threshold* is outside the range ``[-1, 100]``.

    Examples
    --------
    >>> report = missing_value_report(df, threshold=5.0)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__!r}.")
    if not (-1 <= threshold <= 100):
        raise ValueError(f"threshold must be in [-1, 100]; got {threshold}.")

    n = len(df)
    missing_count = df.isna().sum()
    missing_pct = (missing_count / n * 100).round(4) if n else missing_count * 0.0
    present_count = n - missing_count
    present_pct = (100 - missing_pct).round(4)

    report = pd.DataFrame(
        {
            "dtype": df.dtypes,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "present_count": present_count,
            "present_pct": present_pct,
        }
    )

    report = report[report["missing_pct"] > threshold].sort_values(
        "missing_count", ascending=False
    )
    return report


# ---------------------------------------------------------------------------
# 3. Duplicate Analysis
# ---------------------------------------------------------------------------

def duplicate_report(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """Return a DataFrame containing only the duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to inspect.
    subset : list of str, optional
        Column labels to consider when identifying duplicates. ``None``
        (default) uses all columns.
    keep : {'first', 'last', False}, optional
        Determines which duplicates to mark. Passed directly to
        :meth:`pandas.DataFrame.duplicated`. Default ``'first'``.

    Returns
    -------
    pd.DataFrame
        A subset of *df* containing the duplicated rows, preserving the
        original index. An empty DataFrame (with the same columns) is
        returned when no duplicates are found.

    Raises
    ------
    TypeError
        If *df* is not a pandas DataFrame.
    ValueError
        If any column in *subset* is not present in *df*.

    Examples
    --------
    >>> dupes = duplicate_report(df, subset=["customer_id", "order_date"])
    >>> len(dupes)
    42
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__!r}.")

    if subset:
        missing_cols = set(subset) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Columns not found in DataFrame: {sorted(missing_cols)}"
            )

    mask = df.duplicated(subset=subset, keep=keep)
    return df[mask].copy()


# ---------------------------------------------------------------------------
# 4. Datatype Summary
# ---------------------------------------------------------------------------

def datatype_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column dtype information grouped by semantic category.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to inspect.

    Returns
    -------
    pd.DataFrame
        Indexed by column name with columns:

        - ``dtype``     : the pandas dtype string (e.g. ``'int64'``)
        - ``kind``      : numpy kind code (``'i'``, ``'f'``, ``'O'``, ...)
        - ``category``  : human-readable category:
          ``'integer'``, ``'float'``, ``'boolean'``, ``'datetime'``,
          ``'timedelta'``, ``'complex'``, ``'string'``, ``'object'``,
          ``'category'``, or ``'other'``
        - ``nullable``  : ``True`` if the column contains any ``NaN`` / ``NaT``

    Raises
    ------
    TypeError
        If *df* is not a pandas DataFrame.

    Examples
    --------
    >>> summary = datatype_summary(df)
    >>> summary[summary["category"] == "datetime"]
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__!r}.")

    def _category(dtype: object) -> str:
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        if pd.api.types.is_float_dtype(dtype):
            return "float"
        if pd.api.types.is_complex_dtype(dtype):
            return "complex"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        if pd.api.types.is_timedelta64_dtype(dtype):
            return "timedelta"
        # pandas >= 1.0 CategoricalDtype check
        if hasattr(pd.api.types, "is_categorical_dtype"):
            try:
                if pd.api.types.is_categorical_dtype(dtype):  # type: ignore[attr-defined]
                    return "category"
            except TypeError:
                pass
        if isinstance(dtype, pd.CategoricalDtype):
            return "category"
        if pd.api.types.is_string_dtype(dtype):
            return "string" if dtype == object else "object"
        return "other"

    records = []
    for col in df.columns:
        dtype = df[col].dtype
        records.append(
            {
                "column": col,
                "dtype": str(dtype),
                "kind": dtype.kind,
                "category": _category(dtype),
                "nullable": bool(df[col].isna().any()),
            }
        )

    return pd.DataFrame(records).set_index("column")


# ---------------------------------------------------------------------------
# 5. Candidate Primary Key Detection
# ---------------------------------------------------------------------------

def candidate_primary_keys(
    df: pd.DataFrame,
    max_columns: int = 3,
) -> pd.DataFrame:
    """Identify columns (or combinations) that could serve as a primary key.

    A candidate key must satisfy both:

    - **No missing values** in any participating column.
    - **All values are unique** across the combined column set.

    Single-column candidates are always evaluated. Multi-column combinations
    are evaluated up to *max_columns* width to keep runtime manageable.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to inspect.
    max_columns : int, optional
        Maximum number of columns to include in a composite key candidate.
        Default ``3``. Set to ``1`` to restrict to single-column keys only.

    Returns
    -------
    pd.DataFrame
        Each row represents one candidate key with columns:

        - ``key_columns``    : tuple of column names forming the key
        - ``num_columns``    : number of columns in the key
        - ``unique_count``   : number of distinct value combinations
        - ``missing_count``  : total missing values across key columns
        - ``is_unique``      : ``True`` if unique_count == len(df)
        - ``is_complete``    : ``True`` if missing_count == 0

        Rows are ordered by ``(num_columns, is_unique descending)``.

    Raises
    ------
    TypeError
        If *df* is not a pandas DataFrame.
    ValueError
        If *max_columns* is less than 1.

    Examples
    --------
    >>> keys = candidate_primary_keys(df)
    >>> keys[keys["is_unique"] & keys["is_complete"]]
    """
    from itertools import combinations

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__!r}.")
    if max_columns < 1:
        raise ValueError(f"max_columns must be >= 1; got {max_columns}.")

    n = len(df)
    records = []
    max_columns = min(max_columns, len(df.columns))

    for width in range(1, max_columns + 1):
        for cols in combinations(df.columns, width):
            subset = df[list(cols)]
            missing_count = int(subset.isna().sum().sum())
            unique_count = int(subset.drop_duplicates().shape[0])
            records.append(
                {
                    "key_columns": cols,
                    "num_columns": width,
                    "unique_count": unique_count,
                    "missing_count": missing_count,
                    "is_unique": unique_count == n,
                    "is_complete": missing_count == 0,
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    return (
        result
        .sort_values(["num_columns", "is_unique"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 6. Memory Usage
# ---------------------------------------------------------------------------

def memory_usage_report(
    df: pd.DataFrame,
    deep: bool = True,
) -> pd.DataFrame:
    """Return per-column memory usage with optimisation hints.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to inspect.
    deep : bool, optional
        If ``True`` (default), use deep introspection via
        :meth:`pandas.DataFrame.memory_usage` to capture the actual size of
        object-type columns.

    Returns
    -------
    pd.DataFrame
        Indexed by column name with columns:

        - ``dtype``        : the pandas dtype string
        - ``bytes``        : memory consumed by this column in bytes
        - ``kb``           : memory in kilobytes (rounded to 4 dp)
        - ``mb``           : memory in megabytes (rounded to 4 dp)
        - ``pct_of_total`` : share of the DataFrame total memory (0-100)
        - ``hint``         : a brief optimisation suggestion, or ``''``

        A ``__TOTAL__`` row is appended at the bottom.

    Raises
    ------
    TypeError
        If *df* is not a pandas DataFrame.

    Examples
    --------
    >>> mem = memory_usage_report(df)
    >>> mem.loc[mem["mb"] > 10]
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__!r}.")

    usage = df.memory_usage(deep=deep, index=False)
    total_bytes = int(usage.sum())

    def _hint(col: str) -> str:
        dtype = df[col].dtype
        if dtype == object:
            n_unique = df[col].nunique(dropna=False)
            ratio = n_unique / max(len(df), 1)
            if ratio < 0.5:
                return "consider pd.Categorical (low cardinality)"
            return "high-cardinality object; verify if string dtype is needed"
        if pd.api.types.is_integer_dtype(dtype) and str(dtype) == "int64":
            col_min = int(df[col].min())
            col_max = int(df[col].max())
            if col_min >= 0 and col_max <= 255:
                return "downcast to uint8"
            if -128 <= col_min and col_max <= 127:
                return "downcast to int8"
            if -32_768 <= col_min and col_max <= 32_767:
                return "downcast to int16"
            if -2_147_483_648 <= col_min and col_max <= 2_147_483_647:
                return "downcast to int32"
        if str(dtype) == "float64":
            return "consider float32 if precision allows"
        return ""

    records = []
    for col in df.columns:
        col_bytes = int(usage.loc[col])
        records.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "bytes": col_bytes,
                "kb": round(col_bytes / 1_024, 4),
                "mb": round(col_bytes / 1_048_576, 4),
                "pct_of_total": (
                    round(col_bytes / total_bytes * 100, 4) if total_bytes else 0.0
                ),
                "hint": _hint(col),
            }
        )

    report = pd.DataFrame(records).set_index("column")

    total_row = pd.DataFrame(
        [
            {
                "column": "__TOTAL__",
                "dtype": "",
                "bytes": total_bytes,
                "kb": round(total_bytes / 1_024, 4),
                "mb": round(total_bytes / 1_048_576, 4),
                "pct_of_total": 100.0,
                "hint": "",
            }
        ]
    ).set_index("column")

    return pd.concat([report, total_row])


# ---------------------------------------------------------------------------
# 7. Unique Value Counts
# ---------------------------------------------------------------------------

def unique_value_counts(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    top_n: int = 10,
    dropna: bool = False,
) -> pd.DataFrame:
    """Return the top-N most frequent values for each selected column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to inspect.
    columns : list of str, optional
        Columns to analyse. ``None`` (default) analyses all columns.
    top_n : int, optional
        Maximum number of top values to return per column. Default ``10``.
    dropna : bool, optional
        If ``False`` (default), ``NaN`` values are counted and included.

    Returns
    -------
    pd.DataFrame
        A long-format DataFrame with columns:

        - ``column``  : source column name
        - ``value``   : the distinct value
        - ``count``   : frequency count
        - ``pct``     : percentage of total rows
        - ``rank``    : rank within the column (1 = most frequent)

        Rows are sorted by ``(column, rank)``.

    Raises
    ------
    TypeError
        If *df* is not a pandas DataFrame.
    ValueError
        If any column in *columns* is not present in *df*.
    ValueError
        If *top_n* is less than 1.

    Examples
    --------
    >>> counts = unique_value_counts(df, columns=["status", "region"], top_n=5)
    >>> counts[counts["column"] == "status"]
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__!r}.")
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1; got {top_n}.")

    if columns is None:
        columns = list(df.columns)
    else:
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Columns not found in DataFrame: {sorted(missing_cols)}"
            )

    n = len(df)
    records = []

    for col in columns:
        value_counts = df[col].value_counts(dropna=dropna).head(top_n)
        for rank, (value, count) in enumerate(value_counts.items(), start=1):
            records.append(
                {
                    "column": col,
                    "value": value,
                    "count": int(count),
                    "pct": round(count / n * 100, 4) if n else 0.0,
                    "rank": rank,
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    return result.sort_values(["column", "rank"]).reset_index(drop=True)
