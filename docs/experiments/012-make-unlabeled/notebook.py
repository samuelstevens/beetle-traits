import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():

    import matplotlib.pyplot as plt
    import polars as pl
    return pl, plt


@app.cell
def _(pl):
    import marimo as mo

    df = pl.read_parquet(
        "/users/PAS2136/cain429/projects/beetle-traits/docs/experiments/012-make-unlabeled/results/1ykzeqat.parquet"
    )
    return df, mo


@app.cell
def _(df, pl):
    df.filter(pl.col("beetle_id") == "NEON.BET.D14.001520")
    return


@app.cell
def _(df, pl):
    _nested_cols = [
        n for n, t in df.schema.items() if t.base_type() in (pl.Array, pl.List)
    ]
    df.with_columns(
        pl.col(c).cast(pl.List(pl.String)).list.join(",") for c in _nested_cols
    ).write_csv("docs/experiments/012-make-unlabeled/results/1ykzeqat.csv")
    return


@app.cell
def _(df, mo):
    _species = df.get_column("scientific_name").unique().sort().to_list()
    species_picker = mo.ui.dropdown(_species, value=_species[0], label="Species")
    species_picker
    return (species_picker,)


@app.cell
def _(df, mo, pl, plt, species_picker):
    def _():
        import numpy as np
        from PIL import Image

        _species_df = df.filter(pl.col("scientific_name") == species_picker.value)
        _sample_df = _species_df.sample(n=min(20, len(_species_df)), seed=0)
        _n = len(_sample_df)
        _ncols = 4
        _nrows = (_n + _ncols - 1) // _ncols
        _fig, _axes = plt.subplots(
            _nrows,
            _ncols,
            figsize=(4 * _ncols, 4 * _nrows),
            layout="constrained",
            dpi=90,
        )
        _axes_flat = np.asarray(_axes).ravel()
        for _ax in _axes_flat[_n:]:
            _ax.set_visible(False)

        for _ax, _row in zip(_axes_flat, _sample_df.iter_rows(named=True)):
            with Image.open(_row["img_fpath"]) as _fd:
                _img = np.asarray(_fd.convert("RGB"))
            _ax.imshow(_img)

            _pred = np.asarray(_row["pred_coords_px"], dtype=np.float32).reshape(2, 2, 2)
            for _line_i, _col in enumerate(["lime", "yellow"]):
                (_x0, _y0), (_x1, _y1) = _pred[_line_i]
                _ax.plot(
                    [_x0, _x1],
                    [_y0, _y1],
                    color=_col,
                    linewidth=1.8,
                    linestyle="--",
                    marker="x",
                    markersize=5,
                )

            _ax.set_title(_row["beetle_id"] or "", fontsize=7)
            _ax.set_xticks([])
            _ax.set_yticks([])
            _ax.spines[:].set_visible(False)

        _fig.suptitle(
            f"{species_picker.value} -- lime=width pred, yellow=length pred",
            fontsize=10,
        )
        return mo.mpl.interactive(_fig)


    _()
    return


@app.cell
def _(df, mo, pl, plt, species_picker):
    def _():
        import numpy as np
        from PIL import Image

        _target_size = 256
        _species_df = df.filter(pl.col("scientific_name") == species_picker.value)
        _sample_df = _species_df.sample(n=min(20, len(_species_df)), seed=0)
        _n = len(_sample_df)
        _ncols = 4
        _nrows = (_n + _ncols - 1) // _ncols
        _fig, _axes = plt.subplots(
            _nrows,
            _ncols,
            figsize=(4 * _ncols, 4 * _nrows),
            layout="constrained",
            dpi=90,
        )
        _axes_flat = np.asarray(_axes).ravel()
        for _ax in _axes_flat[_n:]:
            _ax.set_visible(False)

        for _ax, _row in zip(_axes_flat, _sample_df.iter_rows(named=True)):
            with Image.open(_row["img_fpath"]) as _fd:
                _orig_w, _orig_h = _fd.size
                _img = np.asarray(
                    _fd.convert("RGB").resize(
                        (_target_size, _target_size), Image.Resampling.BILINEAR
                    )
                )
            _ax.imshow(_img)

            _pred = np.asarray(_row["pred_coords_px"], dtype=np.float32).reshape(2, 2, 2)
            _sx = _target_size / _orig_w
            _sy = _target_size / _orig_h
            for _line_i, _col in enumerate(["lime", "yellow"]):
                (_x0, _y0), (_x1, _y1) = _pred[_line_i]
                _ax.plot(
                    [_x0 * _sx, _x1 * _sx],
                    [_y0 * _sy, _y1 * _sy],
                    color=_col,
                    linewidth=1.8,
                    linestyle="--",
                    marker="x",
                    markersize=5,
                )

            _ax.set_title(_row["beetle_id"] or "", fontsize=7)
            _ax.set_xticks([])
            _ax.set_yticks([])
            _ax.spines[:].set_visible(False)

        _fig.suptitle(
            f"{species_picker.value} @ {_target_size}x{_target_size} -- lime=width pred, yellow=length pred",
            fontsize=10,
        )
        return mo.mpl.interactive(_fig)


    _()
    return


if __name__ == "__main__":
    app.run()
