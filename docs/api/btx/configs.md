Module btx.configs
==================

Functions
---------

`dict_to_dataclass(data: object, cls: type[~T]) ‑> ~T | object`
:   Recursively convert a dictionary to a dataclass instance.

`expand(config: dict[typing.Any, typing.Any]) ‑> Iterator[dict[typing.Any, typing.Any]]`
:   Expand a nested dict that may contain lists into many dicts.

`get_non_default_values(obj: ~T, default_obj: ~T) ‑> dict[typing.Any, typing.Any]`
:   Recursively find fields that differ from defaults.

`load_cfgs(override: ~T, *, default: ~T, sweep_dcts: list[dict[typing.Any, typing.Any]]) ‑> tuple[list[~T], list[str]]`
:   Load a list of configs from a combination of sources.
    
    Args:
        override: Command-line overridden values.
        default: The default values for a config.
        sweep_dcts: A list of dictionaries from Python sweep files. Each dictionary may contain list values that will be expanded.
    
    Returns:
        A list of configs and a list of errors.

`load_sweep(sweep_fpath: pathlib.Path) ‑> list[dict[typing.Any, typing.Any]]`
:   Load a sweep file and return the list of config dicts.
    
    Args:
        sweep_fpath: Path to a Python file with a `make_cfgs()` function.
    
    Returns:
        List of config dictionaries from `make_cfgs()`. Returns empty list if any error occurs.