from __future__ import annotations
from dataclasses import is_dataclass, fields
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    get_args,
    get_origin,
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Annotated,
    Literal,
)
import polars as pl

# --- core scalars ---
_SCALAR_MAP: Dict[type, pl.DataType] = {
    int: pl.Int64,
    float: pl.Float64,
    bool: pl.Boolean,
    str: pl.Utf8,
    datetime: pl.Datetime,
    date: pl.Date,
    time: pl.Time,
    timedelta: pl.Duration,
    bytes: pl.Binary,
    object: pl.Object,
}


def _strip_optional(tp: Any) -> Any:
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(args) != len(get_args(tp)):
            return args[0]
    return tp


def _from_annotated(tp: Any) -> tuple[Any, dict]:
    if get_origin(tp) is Annotated:
        inner, *meta = get_args(tp)
        md = {}
        for m in meta:
            if isinstance(m, dict):
                md.update(m)
            elif isinstance(m, tuple) and m and m[0] == "decimal" and len(m) == 3:
                md["decimal"] = (int(m[1]), int(m[2]))
        return inner, md
    return tp, {}


def _decimal_dtype(md: dict) -> pl.DataType:
    if "decimal" in md:
        p, s = md["decimal"]
        return pl.Decimal(precision=p, scale=s)
    return pl.Decimal(precision=38, scale=9)


def _dataclass_to_struct_dtype(dc_type: Any) -> pl.Struct:
    flds = []
    for f in fields(dc_type):
        flds.append(pl.Field(f.name, _pytype_to_pl_dtype(f.type)))
    return pl.Struct(flds)


def _enum_base_dtype(_: type[Enum]) -> pl.DataType:
    return pl.Utf8  # or pl.Categorical


def _list_dtype(arg: Any) -> pl.List:
    return pl.List(_pytype_to_pl_dtype(arg))


def _tuple_dtype(args: Tuple[Any, ...]) -> pl.List:
    if not args:
        return pl.List(pl.Object)
    first = _pytype_to_pl_dtype(args[0])
    if all(_pytype_to_pl_dtype(a) == first for a in args[1:]):
        return pl.List(first)
    return pl.List(pl.Object)


def _pytype_to_pl_dtype(tp: Any) -> pl.DataType:
    tp, md = _from_annotated(tp)
    tp = _strip_optional(tp)

    if isinstance(tp, type) and is_dataclass(tp):
        return _dataclass_to_struct_dtype(tp)

    if isinstance(tp, type) and issubclass(tp, Enum):
        return _enum_base_dtype(tp)

    if tp in _SCALAR_MAP:
        return _decimal_dtype(md) if tp is Decimal else _SCALAR_MAP[tp]

    origin, args = get_origin(tp), get_args(tp)

    if origin in (list, List, set, frozenset):
        return _list_dtype(args[0] if args else Any)

    if origin in (tuple, Tuple):
        return _tuple_dtype(args)

    if origin in (dict, Dict):
        return pl.Object

    if origin is Literal:
        return _pytype_to_pl_dtype(type(args[0])) if args else pl.Object

    if origin is Union:  # non-optional union
        return pl.Object

    if tp is Decimal:
        return _decimal_dtype(md)

    return pl.Object


# --- public API: return a pl.Struct for the dataclass ---
def dataclass_to_polars_struct(dc_type: Any) -> pl.Struct:
    if not (isinstance(dc_type, type) and is_dataclass(dc_type)):
        raise TypeError("dataclass_to_polars_struct expects a dataclass *type*")
    return _dataclass_to_struct_dtype(dc_type)
