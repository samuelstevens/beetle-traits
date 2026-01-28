import dataclasses
import itertools
import logging
import os
import pathlib
import types
import typing as tp
from collections.abc import Iterator

import beartype

T = tp.TypeVar("T")

logger = logging.getLogger(__name__)


@beartype.beartype
def load_sweep(sweep_fpath: pathlib.Path) -> list[dict[tp.Any, tp.Any]]:
    """
    Load a sweep file and return the list of config dicts.

    Args:
        sweep_fpath: Path to a Python file with a `make_cfgs()` function.

    Returns:
        List of config dictionaries from `make_cfgs()`. Returns empty list if any error occurs.
    """
    try:
        namespace = {}
        exec(sweep_fpath.read_text(), namespace)
        result = namespace["make_cfgs"]()
        if not isinstance(result, list):
            logger.warning(
                f"make_cfgs() in {sweep_fpath} returned {type(result).__name__}, expected list"
            )
            return []
        return result
    except Exception as err:
        logger.warning(f"Failed to load sweep from {sweep_fpath}: {err}")
        return []


@beartype.beartype
def load_cfgs(
    override: T, *, default: T, sweep_dcts: list[dict[tp.Any, tp.Any]]
) -> tuple[list[T], list[str]]:
    """
    Load a list of configs from a combination of sources.

    Args:
        override: Command-line overridden values.
        default: The default values for a config.
        sweep_dcts: A list of dictionaries from Python sweep files. Each dictionary may contain list values that will be expanded.

    Returns:
        A list of configs and a list of errors.
    """
    assert dataclasses.is_dataclass(override) and not isinstance(override, type)
    assert dataclasses.is_dataclass(default) and not isinstance(default, type)

    if not sweep_dcts:
        return [override], []

    overridden_fields = get_non_default_values(override, default)

    cfgs: list[T] = []
    errs: list[str] = []

    d = 0

    for sweep_dct in sweep_dcts:
        filtered_dct = _filter_overridden_fields(sweep_dct, overridden_fields)

        if not filtered_dct:
            cfgs.append(override)
            d += 1
            continue

        try:
            updates = _recursive_dataclass_update(override, filtered_dct, override, d)

            if hasattr(override, "seed") and "seed" not in updates:
                updates["seed"] = getattr(override, "seed", 0) + d

            cfgs.append(dataclasses.replace(override, **updates))
            d += 1
        except Exception as err:
            errs.append(str(err))
            d += 1

    return cfgs, errs


@beartype.beartype
def expand(config: dict[tp.Any, tp.Any]) -> Iterator[dict[tp.Any, tp.Any]]:
    """Expand a nested dict that may contain lists into many dicts."""
    yield from _expand_discrete(dict(config))


@beartype.beartype
def _expand_discrete(config: dict[tp.Any, tp.Any]) -> Iterator[dict[tp.Any, tp.Any]]:
    """Recursively expand lists in a dictionary into multiple dictionaries."""
    if not config:
        yield {}
        return

    key, value = config.popitem()

    if isinstance(value, list):
        for c in _expand_discrete(config):
            for v in value:
                yield {**c, key: v}
    elif isinstance(value, dict):
        for c, v in itertools.product(
            _expand_discrete(config), _expand_discrete(value)
        ):
            yield {**c, key: v}
    else:
        for c in _expand_discrete(config):
            yield {**c, key: value}


@beartype.beartype
def _filter_overridden_fields(
    dct: dict[tp.Any, tp.Any], overridden: dict[tp.Any, tp.Any]
) -> dict[tp.Any, tp.Any]:
    """Remove fields from `dct` that are present in `overridden`."""
    result = {}
    for key, value in dct.items():
        if key not in overridden:
            result[key] = value
            continue

        if isinstance(value, dict):
            overridden_value = overridden.get(key)

            if isinstance(overridden_value, dict):
                filtered = _filter_overridden_fields(value, overridden_value)
                if filtered:
                    result[key] = filtered
                continue

            if dataclasses.is_dataclass(overridden_value) and not isinstance(
                overridden_value, type
            ):
                try:
                    default_value = type(overridden_value)()
                except TypeError:
                    continue

                nested_overrides = get_non_default_values(
                    overridden_value, default_value
                )
                filtered = _filter_overridden_fields(value, nested_overrides)
                if filtered:
                    result[key] = filtered
                continue

            continue

    return result


@beartype.beartype
def _convert_value(value: object, field_type: object) -> object:
    """Convert a value to the correct type based on field_type."""
    origin = tp.get_origin(field_type)
    args = tp.get_args(field_type)

    if origin is tuple and args:
        return tuple(value) if isinstance(value, list) else value
    elif origin is list and args and dataclasses.is_dataclass(args[0]):
        msg = "Expected list for dataclass list field."
        assert isinstance(value, list), msg
        cls = tp.cast(type[T], args[0])
        return [dict_to_dataclass(item, cls) for item in value]
    elif dataclasses.is_dataclass(field_type):
        msg = "Expected dict for dataclass field."
        assert isinstance(value, dict), msg
        cls = tp.cast(type[T], field_type)
        return dict_to_dataclass(value, cls)
    elif field_type is pathlib.Path:
        if value is None:
            return value
        pathlike = tp.cast(str | os.PathLike[str], value)
        return pathlib.Path(pathlike)
    elif origin is tp.Union and pathlib.Path in args:
        if value is None:
            return value
        pathlike = tp.cast(str | os.PathLike[str], value)
        return pathlib.Path(pathlike)
    elif origin is types.UnionType and pathlib.Path in args:
        if value is None:
            return value
        pathlike = tp.cast(str | os.PathLike[str], value)
        return pathlib.Path(pathlike)
    else:
        if isinstance(field_type, type) and not isinstance(value, field_type):
            raise TypeError(
                f"Expected {field_type.__name__}, got {type(value).__name__}"
            )
        return value


@beartype.beartype
def _recursive_dataclass_update(obj, updates: dict[tp.Any, tp.Any], base_cfg, d: int):
    """Recursively update nested dataclasses."""
    if not dataclasses.is_dataclass(obj):
        return updates

    result = {}
    for key, value in updates.items():
        if not hasattr(obj, key):
            result[key] = value
            continue

        attr = getattr(obj, key)
        field_type = type(obj).__dataclass_fields__[key].type

        if dataclasses.is_dataclass(attr) and isinstance(value, dict):
            nested_updates = _recursive_dataclass_update(attr, value, base_cfg, d)

            if hasattr(attr, "seed") and "seed" not in nested_updates:
                base_seed = getattr(base_cfg, "seed", 0) if base_cfg else 0
                nested_updates["seed"] = getattr(attr, "seed", 0) + base_seed + d

            result[key] = dataclasses.replace(attr, **nested_updates)
        else:
            result[key] = _convert_value(value, field_type)

    return result


@beartype.beartype
def dict_to_dataclass(data: object, cls: type[T]) -> T | object:
    """Recursively convert a dictionary to a dataclass instance."""
    if not dataclasses.is_dataclass(cls):
        return data
    msg = "Expected dict for dataclass conversion."
    assert isinstance(data, dict), msg
    data = tp.cast(dict[tp.Any, tp.Any], data)

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}

    for field_name, field_type in field_types.items():
        if field_name not in data:
            continue

        value = data[field_name]

        origin = tp.get_origin(field_type)
        args = tp.get_args(field_type)

        if origin is tuple and args:
            kwargs[field_name] = tuple(value) if isinstance(value, list) else value
        elif origin is list and args and dataclasses.is_dataclass(args[0]):
            msg = "Expected list for dataclass list field."
            assert isinstance(value, list), msg
            cls = tp.cast(type[T], args[0])
            kwargs[field_name] = [dict_to_dataclass(item, cls) for item in value]
        elif dataclasses.is_dataclass(field_type):
            msg = "Expected dict for dataclass field."
            assert isinstance(value, dict), msg
            cls = tp.cast(type[T], field_type)
            kwargs[field_name] = dict_to_dataclass(value, cls)
        elif field_type is pathlib.Path:
            if value is None:
                kwargs[field_name] = value
            else:
                pathlike = tp.cast(str | os.PathLike[str], value)
                kwargs[field_name] = pathlib.Path(pathlike)
        elif origin is tp.Union and pathlib.Path in args:
            if value is None:
                kwargs[field_name] = value
            else:
                pathlike = tp.cast(str | os.PathLike[str], value)
                kwargs[field_name] = pathlib.Path(pathlike)
        elif origin is types.UnionType and pathlib.Path in args:
            if value is None:
                kwargs[field_name] = value
            else:
                pathlike = tp.cast(str | os.PathLike[str], value)
                kwargs[field_name] = pathlib.Path(pathlike)
        else:
            kwargs[field_name] = value

    return cls(**kwargs)


@beartype.beartype
def get_non_default_values(obj: T, default_obj: T) -> dict[tp.Any, tp.Any]:
    """Recursively find fields that differ from defaults."""
    assert dataclasses.is_dataclass(obj) and not isinstance(obj, type)
    assert dataclasses.is_dataclass(default_obj) and not isinstance(default_obj, type)

    diff = {}
    for field in dataclasses.fields(obj):
        obj_value = getattr(obj, field.name)
        default_value = getattr(default_obj, field.name)

        if obj_value == default_value:
            continue

        if (
            dataclasses.is_dataclass(obj_value)
            and dataclasses.is_dataclass(default_value)
            and type(obj_value) is type(default_value)
        ):
            nested_diff = get_non_default_values(obj_value, default_value)
            if nested_diff:
                diff[field.name] = nested_diff
        else:
            diff[field.name] = obj_value

    return diff
