import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import ast
    import pathlib

    import beartype
    import marimo as mo
    import numpy as np
    import polars as pl
    import skimage.feature
    from PIL import Image, ImageDraw

    return Image, ImageDraw, ast, beartype, mo, np, pathlib, pl, skimage


@app.cell
def _(pathlib):
    hf_root = "/fs/scratch/PAS2136/samuelstevens/datasets/beetlepalooza-beetles-main"
    hf_root = pathlib.Path(hf_root)
    assert hf_root.exists()
    return (hf_root,)


@app.cell
def _(pathlib):
    resized_root = (
        "/fs/ess/PAS2136/BeetlePalooza-2024/Resized Images [Corrected from ISA]"
    )
    resized_root = pathlib.Path(resized_root)
    assert resized_root.exists()
    return


@app.cell
def _(pl):
    line_dtype = pl.Struct({
        "x1": pl.Int64,
        "y1": pl.Int64,
        "x2": pl.Int64,
        "y2": pl.Int64,
    })
    return (line_dtype,)


@app.cell
def _(beartype, hf_root, pl):
    @beartype.beartype
    def load_specimens_df() -> pl.DataFrame:
        """Load and process the individual_specimens.csv file."""
        df = pl.read_csv(hf_root / "individual_specimens.csv")

        # Extract beetle position from the individual image filename
        # e.g., "A00000001831_specimen_1.png" -> 1
        df = df.with_columns(
            pl
            .col("groupImageFilePath")
            .str.strip_prefix("group_images/")
            .alias("GroupImgBasename"),
            pl
            .col("individualImageFilePath")
            .str.extract(r"specimen_(\d+)", 1)
            .cast(pl.Int64)
            .alias("BeetlePosition"),
        )

        return df

    specimens_df = load_specimens_df()

    # Look for duplicate_species
    assert (
        specimens_df
        .group_by("GroupImgBasename", "BeetlePosition")
        .len()
        .filter(pl.col("len") > 1)
        .is_empty()
    )
    return (specimens_df,)


@app.cell
def _(specimens_df):
    specimens_df
    return


@app.cell
def _(ast, beartype, hf_root, line_dtype, pl):
    @beartype.beartype
    def load_measurements_df() -> pl.DataFrame:
        """Load and process the BeetleMeasurements.csv file."""
        df = pl.read_csv(hf_root / "BeetleMeasurements.csv")

        # Parse the coords_pix JSON string
        def parse_coords(coords_str):
            coords_str = coords_str.replace('""', '"')
            parsed = ast.literal_eval(coords_str)
            return parsed

        # Process dataframe
        df = (
            df
            .with_columns(
                pl.col("pictureID").alias("GroupImgBasename"),
                pl.col("individual").alias("BeetlePosition"),
                # Tuples
                pl.col("image_dim").map_elements(
                    ast.literal_eval, return_dtype=pl.List(pl.Int64)
                ),
                pl.col("resized_image_dim").map_elements(
                    ast.literal_eval, return_dtype=pl.List(pl.Int64)
                ),
            )
            .filter(pl.col("image_dim").list.sum() > 0)
            .with_columns(
                # Coords
                pl.col("coords_pix").map_elements(
                    parse_coords, return_dtype=line_dtype
                ),
                pl.col("coords_pix_scaled_up").map_elements(
                    parse_coords, return_dtype=line_dtype
                ),
                pl.col("scalebar").map_elements(parse_coords, return_dtype=line_dtype),
            )
        )

        return df

    measurements_df = load_measurements_df()
    measurements_df
    return (measurements_df,)


@app.cell
def _(
    Image,
    ImageDraw,
    hf_root,
    measurements_df,
    mo,
    np,
    pl,
    resized_group_img,
    skimage,
    specimens_df,
):
    def _():
        (group_rel_path, scalebar_coords, resize_tgt) = measurements_df.select(
            "file_name", "scalebar", "resized_image_dim"
        ).row(index=0)

        (indiv_rel_path,) = (
            specimens_df
            .filter(pl.col("groupImageFilePath") == group_rel_path)
            .select("individualImageFilePath")
            .row(index=0)
        )

        main_group_img = Image.open(hf_root / group_rel_path)
        print(group_rel_path)
        # resized_group_img = Image.open(resized_root / pathlib.Path(group_rel_path).name)

        indiv_img = Image.open(hf_root / indiv_rel_path)

        content = []

        draw = ImageDraw.Draw(main_group_img)
        draw.line(
            (
                (scalebar_coords["x1"], scalebar_coords["y1"]),
                (scalebar_coords["x2"], scalebar_coords["y2"]),
            ),
            fill=(0, 255, 0),
            width=12,
        )
        for coords in measurements_df.filter(
            pl.col("file_name") == group_rel_path
        ).get_column("coords_pix"):
            draw.line(
                ((coords["x1"], coords["y1"]), (coords["x2"], coords["y2"])),
                fill=(255, 0, 0),
                width=12,
            )

        content.append(mo.image(main_group_img))
        content.append(mo.image(indiv_img))
        return content

        corr = skimage.feature.match_template(
            np.asarray(main_group_img.convert("L")),
            np.asarray(indiv_img.convert("L")),
            pad_input=False,
        )

        max_corr_idx = np.argmax(corr)
        iy, ix = np.unravel_index(max_corr_idx, corr.shape)
        offset_px = float(ix), float(iy)

        # Get the normalized cross-correlation score at the best match
        ncc_score = float(corr.flat[max_corr_idx])

        print(max_corr_idx)
        print(iy, ix)
        print(offset_px)
        print(ncc_score)

        print("-" * 20)

        corr = skimage.feature.match_template(
            np.asarray(resized_group_img.convert("L")),
            np.asarray(indiv_img.convert("L")),
            pad_input=False,
        )

        max_corr_idx = np.argmax(corr)
        iy, ix = np.unravel_index(max_corr_idx, corr.shape)
        offset_px = float(ix), float(iy)

        # Get the normalized cross-correlation score at the best match
        ncc_score = float(corr.flat[max_corr_idx])

        print(max_corr_idx)
        print(iy, ix)
        print(offset_px)
        print(ncc_score)

        return corr.shape, main_group_img.size, indiv_img.size

        # content.append(mo.image(cropped_img))
        return mo.vstack(content)

    _()
    return


@app.cell
def _(measurements_df):
    measurements_df  # .filter(pl.col("file_name") == group_rel_path)
    return


@app.cell
def _():
    return


@app.cell
def _(Image, ImageDraw, pl):
    def _():
        ann_df = pl.read_json("data/beetlepalooza-formatted/annotations.json")

        path, measurements = ann_df.select("indiv_img_abs_path", "measurements").row(
            index=0
        )

        img = Image.open(path)
        draw = ImageDraw.Draw(img)

        for measurement in measurements:
            coords = measurement["coords_px"]
            print(coords, img.size)
            draw.line(
                (
                    (coords["x1"] - 400, coords["y1"]),
                    (coords["x2"] - 400, coords["y2"]),
                ),
                fill=(255, 0, 0),
                width=12,
            )

        return img

    _()
    return


if __name__ == "__main__":
    app.run()
