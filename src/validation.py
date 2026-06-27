"""
validation.py
=============
Reusable data-validation utilities for the Gyakriti analytics project.

Built for the Olist Brazilian E-Commerce dataset but generic enough to be
reused across any analytics pipeline.

All functions are pure — no side-effects, no printing, no logging.
They return DataFrames, dicts, or booleans so they compose cleanly inside
notebooks, scripts, or dbt-style Python models.

Usage
-----
    import pandas as pd
    from validation import (
        validate_columns,
        validate_dtypes,
        validate_primary_key,
        validate_foreign_key,
        validate_duplicates,
        validate_allowed_values,
        validate_date_order,
        validation_summary,
    )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Column Validation
# ---------------------------------------------------------------------------

def validate_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    allow_extra: bool = True,
) -> dict:
    """Validate that all required columns are present in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    required_columns : list of str
        Column names that must be present.
    allow_extra : bool, optional
        If ``False``, columns present in *df* but not in *required_columns*
        are flagged as unexpected. Default ``True``.

    Returns
    -------
    dict
        Keys:

        - ``"required"``    : list of required column names
        - ``"actual"``      : list of actual column names in *df*
        - ``"missing"``     : required columns absent from *df*
        - ``"unexpected"``  : columns in *df* not in *required_columns*
                              (empty list when *allow_extra* is ``True``)
        - ``"passed"``      : ``True`` only when there are no missing columns
                              (and no unexpected columns if *allow_extra* is False)

    Examples
    --------
    >>> result = validate_columns(df, ["order_id", "customer_id"])
    >>> result["passed"]
    True
    """
    required = list(required_columns)
    actual = list(df.columns)
    missing = [c for c in required if c not in actual]
    unexpected = [] if allow_extra else [c for c in actual if c not in required]

    passed = len(missing) == 0 and len(unexpected) == 0

    return {
        "required": required,
        "actual": actual,
        "missing": missing,
        "unexpected": unexpected,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# 2. Data Type Validation
# ---------------------------------------------------------------------------

def validate_dtypes(
    df: pd.DataFrame,
    expected_dtypes: Dict[str, str],
) -> pd.DataFrame:
    """Compare expected data types against actual column dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    expected_dtypes : dict
        Mapping of ``{column_name: expected_dtype_string}``.
        The dtype string is compared against ``str(df[col].dtype)``.
        Example: ``{"order_id": "object", "price": "float64"}``.

    Returns
    -------
    pd.DataFrame
        One row per column in *expected_dtypes* with columns:

        - ``column``         : column name
        - ``expected_dtype`` : dtype you specified
        - ``actual_dtype``   : dtype found in *df* (``"MISSING"`` if absent)
        - ``match``          : ``True`` when expected == actual

    Examples
    --------
    >>> schema = {"order_id": "object", "freight_value": "float64"}
    >>> report = validate_dtypes(df, schema)
    >>> report[~report["match"]]
    """
    records = []
    for col, expected in expected_dtypes.items():
        if col in df.columns:
            actual = str(df[col].dtype)
        else:
            actual = "MISSING"
        records.append(
            {
                "column": col,
                "expected_dtype": expected,
                "actual_dtype": actual,
                "match": expected == actual,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Primary Key Validation
# ---------------------------------------------------------------------------

def validate_primary_key(
    df: pd.DataFrame,
    key_columns: Union[str, List[str]],
) -> dict:
    """Check whether one or more columns can act as a primary key.

    A valid primary key must be both unique and free of null values across
    all participating columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    key_columns : str or list of str
        Single column name or list of column names forming the composite key.

    Returns
    -------
    dict
        Keys:

        - ``"key_columns"``    : the key column(s) evaluated
        - ``"row_count"``      : total rows in *df*
        - ``"unique_count"``   : number of distinct key value combinations
        - ``"duplicate_count"``: number of rows that are duplicates
        - ``"null_count"``     : total null values across key columns
        - ``"is_unique"``      : ``True`` when no duplicates exist
        - ``"is_complete"``    : ``True`` when no nulls exist
        - ``"passed"``         : ``True`` when both unique and complete
        - ``"duplicate_rows"`` : DataFrame of the actual duplicate rows

    Raises
    ------
    ValueError
        If any column in *key_columns* is not present in *df*.

    Examples
    --------
    >>> result = validate_primary_key(orders, "order_id")
    >>> result["passed"]
    True
    """
    if isinstance(key_columns, str):
        key_columns = [key_columns]

    missing_cols = [c for c in key_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Key columns not found in DataFrame: {missing_cols}")

    subset = df[key_columns]
    row_count = len(df)
    null_count = int(subset.isna().sum().sum())
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)
    duplicate_rows = df[duplicate_mask].copy()
    duplicate_count = int(duplicate_mask.sum())
    unique_count = int(subset.drop_duplicates().shape[0])
    is_unique = duplicate_count == 0
    is_complete = null_count == 0

    return {
        "key_columns": key_columns,
        "row_count": row_count,
        "unique_count": unique_count,
        "duplicate_count": duplicate_count,
        "null_count": null_count,
        "is_unique": is_unique,
        "is_complete": is_complete,
        "passed": is_unique and is_complete,
        "duplicate_rows": duplicate_rows,
    }


# ---------------------------------------------------------------------------
# 4. Foreign Key Validation
# ---------------------------------------------------------------------------

def validate_foreign_key(
    child_df: pd.DataFrame,
    child_col: str,
    parent_df: pd.DataFrame,
    parent_col: str,
) -> dict:
    """Validate referential integrity between two tables.

    Checks that every value in ``child_df[child_col]`` exists in
    ``parent_df[parent_col]``. Null values in the child column are excluded
    from the orphan check but counted separately.

    Parameters
    ----------
    child_df : pd.DataFrame
        The referencing (child) table, e.g. ``orders``.
    child_col : str
        The foreign-key column in the child table, e.g. ``"customer_id"``.
    parent_df : pd.DataFrame
        The referenced (parent) table, e.g. ``customers``.
    parent_col : str
        The primary-key column in the parent table, e.g. ``"customer_id"``.

    Returns
    -------
    dict
        Keys:

        - ``"child_col"``          : foreign-key column name
        - ``"parent_col"``         : parent primary-key column name
        - ``"child_row_count"``    : total rows in child table
        - ``"parent_key_count"``   : distinct values in parent key column
        - ``"null_count"``         : null values in child foreign-key column
        - ``"orphan_count"``       : child rows whose key is absent in parent
        - ``"orphan_rows"``        : DataFrame of the orphaned child rows
        - ``"missing_keys"``       : list of key values absent in parent
        - ``"passed"``             : ``True`` when orphan_count == 0

    Raises
    ------
    ValueError
        If *child_col* is not in *child_df* or *parent_col* is not in *parent_df*.

    Examples
    --------
    >>> result = validate_foreign_key(orders, "customer_id", customers, "customer_id")
    >>> result["passed"]
    True
    """
    if child_col not in child_df.columns:
        raise ValueError(f"Column '{child_col}' not found in child DataFrame.")
    if parent_col not in parent_df.columns:
        raise ValueError(f"Column '{parent_col}' not found in parent DataFrame.")

    parent_keys = set(parent_df[parent_col].dropna().unique())
    child_series = child_df[child_col]

    null_count = int(child_series.isna().sum())
    non_null_mask = child_series.notna()
    orphan_mask = non_null_mask & ~child_series.isin(parent_keys)
    orphan_rows = child_df[orphan_mask].copy()
    missing_keys = sorted(child_series[orphan_mask].unique().tolist())

    return {
        "child_col": child_col,
        "parent_col": parent_col,
        "child_row_count": len(child_df),
        "parent_key_count": len(parent_keys),
        "null_count": null_count,
        "orphan_count": len(orphan_rows),
        "orphan_rows": orphan_rows,
        "missing_keys": missing_keys,
        "passed": len(orphan_rows) == 0,
    }


# ---------------------------------------------------------------------------
# 5. Duplicate Validation
# ---------------------------------------------------------------------------

def validate_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> dict:
    """Return duplicate rows in a DataFrame along with summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    subset : list of str, optional
        Columns used to identify duplicates. ``None`` (default) considers
        all columns.
    keep : {'first', 'last', False}, optional
        Which occurrence to mark as a duplicate. Default ``'first'``.
        Pass ``False`` to flag every copy of a duplicated record.

    Returns
    -------
    dict
        Keys:

        - ``"total_rows"``       : total rows in *df*
        - ``"duplicate_count"``  : number of duplicate rows flagged
        - ``"duplicate_pct"``    : duplicate rows as a percentage of total
        - ``"subset"``           : columns used in deduplication check
        - ``"passed"``           : ``True`` when no duplicates found
        - ``"duplicate_rows"``   : DataFrame of the duplicate rows

    Raises
    ------
    ValueError
        If any column in *subset* is not present in *df*.

    Examples
    --------
    >>> result = validate_duplicates(orders, subset=["order_id"])
    >>> result["duplicate_rows"]
    """
    if subset:
        missing_cols = [c for c in subset if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    mask = df.duplicated(subset=subset, keep=keep)
    duplicate_rows = df[mask].copy()
    total_rows = len(df)
    duplicate_count = int(mask.sum())
    duplicate_pct = round(duplicate_count / total_rows * 100, 4) if total_rows else 0.0

    return {
        "total_rows": total_rows,
        "duplicate_count": duplicate_count,
        "duplicate_pct": duplicate_pct,
        "subset": subset if subset else list(df.columns),
        "passed": duplicate_count == 0,
        "duplicate_rows": duplicate_rows,
    }


# ---------------------------------------------------------------------------
# 6. Allowed Values Validation
# ---------------------------------------------------------------------------

def validate_allowed_values(
    df: pd.DataFrame,
    column: str,
    allowed_values: List,
    include_nulls: bool = False,
) -> dict:
    """Check that a categorical column contains only permitted values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    column : str
        The column whose values are being validated.
    allowed_values : list
        The exhaustive set of permitted values.
    include_nulls : bool, optional
        If ``True``, treat null values as invalid. Default ``False``
        (nulls are excluded from the validity check).

    Returns
    -------
    dict
        Keys:

        - ``"column"``          : column name validated
        - ``"allowed_values"``  : the permitted value set
        - ``"invalid_values"``  : list of values found that are not allowed
        - ``"invalid_count"``   : number of rows with an invalid value
        - ``"invalid_pct"``     : percentage of total rows that are invalid
        - ``"passed"``          : ``True`` when no invalid values are found
        - ``"invalid_rows"``    : DataFrame of rows containing invalid values

    Raises
    ------
    ValueError
        If *column* is not present in *df*.

    Examples
    --------
    >>> allowed = ["created", "approved", "delivered", "cancelled", "shipped"]
    >>> result = validate_allowed_values(orders, "order_status", allowed)
    >>> result["invalid_values"]
    []
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    allowed_set = set(allowed_values)
    series = df[column]

    if not include_nulls:
        series = series.dropna()
        mask = ~df[column].isna() & ~df[column].isin(allowed_set)
    else:
        mask = ~df[column].isin(allowed_set)

    invalid_rows = df[mask].copy()
    invalid_values = sorted(
        invalid_rows[column].dropna().unique().tolist(),
        key=str,
    )
    total_rows = len(df)
    invalid_count = len(invalid_rows)
    invalid_pct = round(invalid_count / total_rows * 100, 4) if total_rows else 0.0

    return {
        "column": column,
        "allowed_values": list(allowed_values),
        "invalid_values": invalid_values,
        "invalid_count": invalid_count,
        "invalid_pct": invalid_pct,
        "passed": invalid_count == 0,
        "invalid_rows": invalid_rows,
    }


# ---------------------------------------------------------------------------
# 7. Date Order Validation
# ---------------------------------------------------------------------------

def validate_date_order(
    df: pd.DataFrame,
    earlier_col: str,
    later_col: str,
    allow_equal: bool = True,
) -> dict:
    """Validate that one datetime column occurs before (or on) another.

    Rows where either column is null are excluded from comparison but counted
    separately.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    earlier_col : str
        Column that must contain the earlier (or equal) datetime.
        Example: ``"purchase_timestamp"``.
    later_col : str
        Column that must contain the later (or equal) datetime.
        Example: ``"delivery_date"``.
    allow_equal : bool, optional
        If ``True`` (default), rows where both timestamps are identical are
        considered valid. Set to ``False`` to require a strict earlier-than
        relationship.

    Returns
    -------
    dict
        Keys:

        - ``"earlier_col"``    : name of the earlier column
        - ``"later_col"``      : name of the later column
        - ``"total_rows"``     : total rows in *df*
        - ``"null_rows"``      : rows skipped due to nulls in either column
        - ``"violation_count"``: rows where the date order is violated
        - ``"violation_pct"``  : violations as a percentage of comparable rows
        - ``"passed"``         : ``True`` when no violations are found
        - ``"violation_rows"`` : DataFrame of rows that violate the rule

    Raises
    ------
    ValueError
        If *earlier_col* or *later_col* is not present in *df*.

    Examples
    --------
    >>> result = validate_date_order(
    ...     orders,
    ...     earlier_col="order_purchase_timestamp",
    ...     later_col="order_delivered_customer_date",
    ... )
    >>> result["passed"]
    True
    """
    for col in (earlier_col, later_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    comparable = df[[earlier_col, later_col]].copy()
    null_mask = comparable[earlier_col].isna() | comparable[later_col].isna()
    null_rows = int(null_mask.sum())
    valid_df = comparable[~null_mask]

    if allow_equal:
        violation_mask = valid_df[earlier_col] > valid_df[later_col]
    else:
        violation_mask = valid_df[earlier_col] >= valid_df[later_col]

    violation_idx = valid_df[violation_mask].index
    violation_rows = df.loc[violation_idx].copy()
    violation_count = len(violation_rows)
    comparable_rows = len(valid_df)
    violation_pct = (
        round(violation_count / comparable_rows * 100, 4) if comparable_rows else 0.0
    )

    return {
        "earlier_col": earlier_col,
        "later_col": later_col,
        "total_rows": len(df),
        "null_rows": null_rows,
        "violation_count": violation_count,
        "violation_pct": violation_pct,
        "passed": violation_count == 0,
        "violation_rows": violation_rows,
    }


# ---------------------------------------------------------------------------
# 8. Validation Summary
# ---------------------------------------------------------------------------

def validation_summary(results: Dict[str, dict]) -> pd.DataFrame:
    """Consolidate multiple validation results into a single summary table.

    Accepts the dict outputs of any validation function in this module and
    produces a concise pass/fail report suitable for display in a notebook or
    final analytics report.

    Parameters
    ----------
    results : dict
        Mapping of ``{validation_name: result_dict}``, where each
        ``result_dict`` is the output of a validation function.  The only
        key that every result dict *must* contain is ``"passed"`` (bool).
        Optional keys used when present:

        - ``"missing"``          -> issue count (column validation)
        - ``"duplicate_count"``  -> issue count (duplicate / primary-key)
        - ``"orphan_count"``     -> issue count (foreign-key)
        - ``"invalid_count"``    -> issue count (allowed-values)
        - ``"violation_count"``  -> issue count (date-order)
        - ``"null_count"``       -> appended to notes when non-zero

    Returns
    -------
    pd.DataFrame
        One row per named validation with columns:

        - ``"validation"``   : name provided in *results* dict key
        - ``"status"``       : ``"PASS"`` or ``"FAIL"``
        - ``"issues"``       : number of problems detected (0 for a pass)
        - ``"notes"``        : brief human-readable explanation

    Examples
    --------
    >>> summary = validation_summary({
    ...     "orders_pk": validate_primary_key(orders, "order_id"),
    ...     "fk_customer": validate_foreign_key(orders, "customer_id",
    ...                                         customers, "customer_id"),
    ... })
    >>> summary
    """
    def _issue_count(result: dict) -> int:
        for key in (
            "missing",
            "duplicate_count",
            "orphan_count",
            "invalid_count",
            "violation_count",
        ):
            if key in result:
                val = result[key]
                # "missing" may be a list
                return len(val) if isinstance(val, list) else int(val)
        return 0

    def _notes(name: str, result: dict) -> str:
        passed = result.get("passed", False)
        issues = _issue_count(result)

        if passed:
            return "All checks passed."

        parts = []

        if "missing" in result and result["missing"]:
            parts.append(f"Missing columns: {result['missing']}")
        if "unexpected" in result and result["unexpected"]:
            parts.append(f"Unexpected columns: {result['unexpected']}")
        if "duplicate_count" in result and result["duplicate_count"]:
            parts.append(f"{result['duplicate_count']} duplicate row(s)")
        if "null_count" in result and result["null_count"]:
            parts.append(f"{result['null_count']} null value(s) in key")
        if "orphan_count" in result and result["orphan_count"]:
            missing_keys = result.get("missing_keys", [])
            sample = missing_keys[:3]
            parts.append(
                f"{result['orphan_count']} orphan row(s); "
                f"missing keys (sample): {sample}"
            )
        if "invalid_values" in result and result["invalid_values"]:
            sample = result["invalid_values"][:5]
            parts.append(f"Invalid values (sample): {sample}")
        if "violation_count" in result and result["violation_count"]:
            parts.append(
                f"{result['violation_count']} date-order violation(s) "
                f"({result.get('violation_pct', 0):.2f}%)"
            )

        return "; ".join(parts) if parts else f"{issues} issue(s) found."

    records = []
    for validation_name, result in results.items():
        passed = bool(result.get("passed", False))
        records.append(
            {
                "validation": validation_name,
                "status": "PASS" if passed else "FAIL",
                "issues": _issue_count(result),
                "notes": _notes(validation_name, result),
            }
        )

    return pd.DataFrame(records)
