import pathlib

from btx import configs


def test_exp004_heatmap_sweep_uses_ce_loss():
    sweep_fpath = (
        pathlib.Path(__file__).resolve().parents[1]
        / "docs"
        / "experiments"
        / "004-heatmap"
        / "sweep.py"
    )
    sweep_dcts = configs.load_sweep(sweep_fpath)

    assert len(sweep_dcts) == 12
    for sweep_dct in sweep_dcts:
        assert sweep_dct["objective"]["heatmap_loss"] == "ce"
