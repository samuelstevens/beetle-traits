import dataclasses
import pathlib

import beartype

from btx import configs


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ObjectiveConfig:
    sparsity_coeff: float = 1e-3


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-4
    objective: ObjectiveConfig = dataclasses.field(default_factory=ObjectiveConfig)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class NestedConfig:
    inner_value: int = 1


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ConfigWithNested:
    nested: NestedConfig = dataclasses.field(default_factory=NestedConfig)
    outer_value: int = 5


def test_dict_to_dataclass_with_missing_fields():
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Config:
        a: int = 1
        b: int = 2

    result = configs.dict_to_dataclass({"a": 10}, Config)

    assert result.a == 10
    assert result.b == 2


def test_dict_to_dataclass_non_dataclass():
    result = configs.dict_to_dataclass({"a": 1}, dict)

    assert result == {"a": 1}


def test_load_cfgs_all_nested_fields_overridden():
    override = ConfigWithNested(nested=NestedConfig(inner_value=999))
    default = ConfigWithNested()
    sweep_dcts = [
        {"nested": {"inner_value": 1}, "outer_value": 10},
        {"nested": {"inner_value": 2}, "outer_value": 10},
        {"nested": {"inner_value": 3}, "outer_value": 10},
        {"nested": {"inner_value": 1}, "outer_value": 20},
        {"nested": {"inner_value": 2}, "outer_value": 20},
        {"nested": {"inner_value": 3}, "outer_value": 20},
    ]

    cfgs, errs = configs.load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert not errs
    assert len(cfgs) == 6
    assert all(cfg.nested.inner_value == 999 for cfg in cfgs)
    assert cfgs[0].outer_value == 10
    assert cfgs[1].outer_value == 10


def test_load_cfgs_from_python_sweep():
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "simple.py"
    sweep_dcts = configs.load_sweep(sweep_fpath)

    override = TrainConfig()
    default = TrainConfig()
    cfgs, errs = configs.load_cfgs(
        override, default=default, sweep_dcts=[sweep_dcts[0]]
    )

    assert len(cfgs) == 1
    assert isinstance(cfgs[0], TrainConfig)
    assert cfgs[0].lr == 1e-4
    assert cfgs[0].objective.sparsity_coeff == 4e-4
    assert len(errs) == 0


def test_load_sweep_missing_function():
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "no_function.py"
    result = configs.load_sweep(sweep_fpath)

    assert result == []


def test_load_sweep_raises_error():
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "raises_error.py"
    result = configs.load_sweep(sweep_fpath)

    assert result == []


def test_load_sweep_wrong_return_type():
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "wrong_return_type.py"
    result = configs.load_sweep(sweep_fpath)

    assert result == []


def test_load_sweep_empty():
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "empty.py"
    sweep_dcts = configs.load_sweep(sweep_fpath)

    assert sweep_dcts == []


def test_load_sweep_with_imports():
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "with_imports.py"
    sweep_dcts = configs.load_sweep(sweep_fpath)

    assert len(sweep_dcts) == 4
    assert sweep_dcts[0] == {"lr": 1e-4, "sparsity": 4e-4}
    assert sweep_dcts[1] == {"lr": 1e-4, "sparsity": 8e-4}
    assert sweep_dcts[2] == {"lr": 3e-4, "sparsity": 4e-4}
    assert sweep_dcts[3] == {"lr": 3e-4, "sparsity": 8e-4}


def test_load_sweep_invalid_syntax():
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "invalid_syntax.py"
    result = configs.load_sweep(sweep_fpath)

    assert result == []


def test_load_sweep_nonexistent_file():
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "does_not_exist.py"
    result = configs.load_sweep(sweep_fpath)

    assert result == []
