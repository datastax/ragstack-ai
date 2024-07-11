import json
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, cast


class MetadataIndexingMode(Enum):
    DEFAULT_TO_UNSEARCHABLE = 1
    DEFAULT_TO_SEARCHABLE = 2


MetadataIndexingPolicy = Tuple[MetadataIndexingMode, Set[str]]


def _normalize_metadata_indexing_policy(
        metadata_indexing: Union[Tuple[str, Iterable[str]], str]
    ) -> MetadataIndexingPolicy:
    mode: MetadataIndexingMode
    fields: Set[str]
    # metadata indexing policy normalization:
    if isinstance(metadata_indexing, str):
        if metadata_indexing.lower() == "all":
            mode, fields = (MetadataIndexingMode.DEFAULT_TO_SEARCHABLE, set())
        elif metadata_indexing.lower() == "none":
            mode, fields = (MetadataIndexingMode.DEFAULT_TO_UNSEARCHABLE, set())
        else:
            raise ValueError(
                f"Unsupported metadata_indexing value '{metadata_indexing}'"
            )
    else:
        assert len(metadata_indexing) == 2
        # it's a 2-tuple (mode, fields) still to normalize
        _mode, _field_spec = metadata_indexing
        fields = {_field_spec} if isinstance(_field_spec, str) else set(_field_spec)
        if _mode.lower() in {
            "default_to_unsearchable",
            "allowlist",
            "allow",
            "allow_list",
        }:
            mode = MetadataIndexingMode.DEFAULT_TO_UNSEARCHABLE
        elif _mode.lower() in {
            "default_to_searchable",
            "denylist",
            "deny",
            "deny_list",
        }:
            mode = MetadataIndexingMode.DEFAULT_TO_SEARCHABLE
        else:
            raise ValueError(
                f"Unsupported metadata indexing mode specification '{_mode}'"
            )
    return (mode, fields)

def _serialize_md_dict(md_dict: Dict[str, Any]) -> str:
    return json.dumps(md_dict, separators=(",", ":"), sort_keys=True)

def _deserialize_md_dict(md_string: str) -> Dict[str, Any]:
    return cast(Dict[str, Any], json.loads(md_string))

def _coerce_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    elif isinstance(value, bool):
        # bool MUST come before int in this chain of ifs!
        return json.dumps(value)
    elif isinstance(value, int):
        # we don't want to store '1' and '1.0' differently
        # for the sake of metadata-filtered retrieval:
        return json.dumps(float(value))
    elif isinstance(value, float):
        return json.dumps(value)
    elif value is None:
        return json.dumps(value)
    else:
        # when all else fails ...
        return str(value)

def _split_metadata_fields(md_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Split the *indexed* part of the metadata in separate parts,
    one per Cassandra column.

    Currently: everything gets cast to a string and goes to a single table
    column. This means:
        - strings are fine
        - floats and integers v: they are cast to str(v)
        - booleans: 'true'/'false' (JSON style)
        - None => 'null' (JSON style)
        - anything else v => str(v), no questions asked

    Caveat: one gets strings back when reading metadata
    """

    # TODO: more care about types here
    stringy_part = {k: _coerce_string(v) for k, v in md_dict.items()}
    return {
        "metadata_s": stringy_part,
    }

def _normalize_row(raw_row: Any) -> Dict[str, Any]:
    md_columns_defaults: Dict[str, Any] = {
        "metadata_s": {},
    }
    pre_normalized = raw_row

    row_rest = {
        k: v
        for k, v in pre_normalized.items()
        if k not in md_columns_defaults
        if k != "attributes_blob"
    }
    merge_md_fields = {
        k: v for k, v in pre_normalized.items() if k in md_columns_defaults
    }
    normalized_merge_md_fields = {
        k: v if v is not None else md_columns_defaults[k]
        for k, v in merge_md_fields.items()
    }
    r_md_from_s = {
        k: v for k, v in normalized_merge_md_fields["metadata_s"].items()
    }

    raw_attr_blob = pre_normalized.get("attributes_blob")
    if raw_attr_blob is not None:
        r_attrs = _deserialize_md_dict(raw_attr_blob)
    else:
        r_attrs = {}

    row_metadata = {
        "metadata": {
            **r_attrs,
            **r_md_from_s,
        },
    }

    normalized = {
        **row_metadata,
        **row_rest,
    }
    return normalized

def _normalize_kwargs(metadata_indexing_policy: Any, args_dict: Dict[str, Any]) -> Dict[str, Any]:
    _metadata_input_dict = args_dict.get("metadata", {})
    # separate indexed and non-indexed (=attributes) as per indexing policy
    metadata_indexed_dict = {
        k: v
        for k, v in _metadata_input_dict.items()
        if is_metadata_field_indexed(k, metadata_indexing_policy)
    }
    attributes_dict = {
        k: _coerce_string(v)
        for k, v in _metadata_input_dict.items()
        if not is_metadata_field_indexed(k, metadata_indexing_policy)
    }

    if attributes_dict != {}:
        attributes_fields = {
            "attributes_blob": _serialize_md_dict(attributes_dict)
        }
    else:
        attributes_fields = {}

    new_metadata_fields = {
        k: v
        for k, v in _split_metadata_fields(metadata_indexed_dict).items()
        if v != {} and v != set()
    }

    new_args_dict = {
        **{k: v for k, v in args_dict.items() if k != "metadata"},
        **attributes_fields,
        **new_metadata_fields,
    }
    return new_args_dict

def _extract_where_clause_blocks(args_dict: Any) -> Tuple[Any, List[str], Tuple[Any, ...]]:
    # This always happens after a corresponding _normalize_kwargs,
    # so the metadata, if present, appears as split-fields.
    assert "metadata" not in args_dict
    if "attributes_blob" in args_dict:
        raise ValueError("Non-indexed metadata fields cannot be used in queries.")
    md_keys = {"metadata_s"}
    # Here the "metadata" entry is made into specific where clauses
    split_metadata = {k: v for k, v in args_dict.items() if k in md_keys}
    these_wc_blocks: List[str] = []
    these_wc_vals_list: List[Any] = []
    # WHERE creation:
    for k, v in sorted(split_metadata.get("metadata_s", {}).items()):
        these_wc_blocks.append(f"metadata_s['{k}'] = %s")
        these_wc_vals_list.append(v)
    # no new kwargs keys are created, all goes to WHERE
    this_args_dict: Dict[str, Any] = {}
    these_wc_vals = tuple(these_wc_vals_list)
    return (
        this_args_dict,
        these_wc_blocks,
        tuple(list(these_wc_vals)),
    )

def _get_find_entries_cql(n: int, **kwargs: Any) -> Tuple[str, Tuple[Any, ...]]:
    columns_desc, where_clause, get_cql_vals = _parse_select_core_params(
        **kwargs
    )
    limit_clause = "LIMIT %s"
    limit_cql_vals = [n]
    select_vals = tuple(list(get_cql_vals) + limit_cql_vals)
    #
    select_cql = SELECT_CQL_TEMPLATE.format(
        columns_desc=columns_desc,
        where_clause=where_clause,
        limit_clause=limit_clause,
    )
    return select_cql, select_vals

def find_entries(self, n: int, **kwargs: Any) -> Iterable[RowType]:
    select_cql, select_vals = self._get_find_entries_cql(n, **kwargs)
    result_set = self.execute_cql(
        select_cql, args=select_vals, op_type=CQLOpType.READ
    )
    return (self._normalize_row(result) for result in result_set)

def _get_to_delete_and_visited(
    n: Optional[int],
    batch_size: int,
    visited_tuples: Set[Tuple[Any, ...]],
    del_pkargs: Optional[List[Any]] = None,
) -> Tuple[int, Set[Tuple[Any, ...]]]:
    if del_pkargs is not None:
        visited_tuples.update(tuple(del_pkarg) for del_pkarg in del_pkargs)
    if n is not None:
        to_delete = min(batch_size, n - len(visited_tuples))
    else:
        to_delete = batch_size
    return to_delete, visited_tuples

def find_and_delete_entries(n: Optional[int] = None, batch_size: int = 20, **kwargs: Any) -> int:
    # Use `find_entries` to delete entries based
    # on queries with metadata, etc. Suitable when `find_entries` is a fit.
    # Returns the number of rows supposedly deleted.
    # Warning: reads before writing. Not very efficient (nor Cassandraic).
    #
    # TODO: Use the 'columns' for a narrowed projection
    # TODO: decouple finding and deleting (streaming) for faster performance
    primary_key_cols = [col for col, _ in _schema_primary_key()]
    #
    batch_size = 20
    to_delete, visited_tuples = _get_to_delete_and_visited(
        n, batch_size, set()
    )
    while to_delete > 0:
        del_pkargs = [
            [found_row[pkc] for pkc in primary_key_cols]
            for found_row in find_entries(n=to_delete, **kwargs)
        ]
        if del_pkargs == []:
            break
        d_futures = [
            delete_async(
                **{pkc: pkv for pkc, pkv in zip(primary_key_cols, del_pkarg)}
            )
            for del_pkarg in del_pkargs
            if tuple(del_pkarg) not in visited_tuples
        ]
        if d_futures == []:
            break
        for d_future in d_futures:
            _ = d_future.result()
        to_delete, visited_tuples = _get_to_delete_and_visited(
            n, batch_size, visited_tuples, del_pkargs
        )
    #
    return len(visited_tuples)
