import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import collections
    import json
    import pathlib

    import beartype
    import marimo as mo
    import numpy as np
    import polars as pl
    import skimage.feature
    from jaxtyping import Float
    from PIL import Image, ImageDraw

    return (
        Float,
        Image,
        ImageDraw,
        beartype,
        collections,
        json,
        mo,
        np,
        pathlib,
        pl,
        skimage,
    )


@app.cell
def _(pathlib):
    hf_root = "/fs/scratch/PAS2136/samuelstevens/datasets/hawaii-beetles"
    hf_root = pathlib.Path(hf_root)
    assert hf_root.exists()
    return (hf_root,)


@app.cell
def _(hf_root, pl):
    trait_df = pl.read_csv(hf_root / "trait_annotations.csv").with_columns(
        pl.col("groupImageFilePath")
        .str.to_lowercase()
        .str.strip_prefix("group_images/")
        .str.strip_suffix(".png")
        .alias("GroupImgBasename"),
        pl.col("coords_scalebar").str.json_decode(),
        pl.col("coords_elytra_max_length").str.json_decode(),
        pl.col("coords_basal_pronotum_width").str.json_decode(),
        pl.col("coords_elytra_max_width").str.json_decode(),
    )
    trait_df.group_by("GroupImgBasename", "BeetlePosition").len().filter(
        pl.col("len") > 1
    ).is_empty()
    return (trait_df,)


@app.cell
def _(hf_root, pl):
    img_df = (
        pl.read_csv(hf_root / "images_metadata.csv").with_columns(
            pl.col("groupImageFilePath")
            .str.to_lowercase()
            .str.strip_prefix("group_images/")
            .str.strip_suffix(".png")
            .alias("GroupImgBasename"),
            pl.col("individualImageFilePath")
            .str.to_lowercase()
            .str.extract(r"specimen_(\d+)", 1)
            .cast(pl.Int64)
            .alias("BeetlePosition"),
        )
        # .select(
        #     "GroupImgBasename", "BeetlePosition", "individualImageFilePath", "individualID"
        # )
    )
    img_df
    return (img_df,)


@app.cell
def _(pl):
    anns_df = pl.read_json(
        "/fs/ess/PAS2136/Hawaii-2025/beetles_intake/BeetlePUUM/Annotations/HawaiiBeetles_Measurements.json"
    ).with_columns(
        pl.col("toras_path")
        .str.to_lowercase()
        .str.strip_prefix("/")
        .str.strip_suffix(".jpg")
        .alias("GroupImgBasename")
    )
    anns_df
    return (anns_df,)


@app.cell
def _(anns_df, img_df, trait_df):
    trait_df.join(anns_df, on=["GroupImgBasename", "BeetlePosition"], how="inner").join(
        img_df, on=["GroupImgBasename", "BeetlePosition"], how="inner"
    ).columns
    return


@app.cell
def _(Float, Image, beartype, np, pathlib):
    @beartype.beartype
    def img_as_arr(
        img: Image.Image | str | pathlib.Path,
    ) -> Float[np.ndarray, "width height channels"]:
        img = img if isinstance(img, Image.Image) else Image.open(img)
        return np.asarray(img, dtype=np.float32)

    return (img_as_arr,)


@app.cell
def _(Image, beartype, hf_root, img_as_arr, img_df, np, pathlib, skimage):
    @beartype.beartype
    def find_origin(
        group_img: Image.Image | str | pathlib.Path,
        indiv_img: Image.Image | str | pathlib.Path,
    ) -> tuple[int, int]:
        """Find where the individual image sits inside the group image.
        Returns (x0, y0) in GROUP pixels (top-left of the individual crop)."""
        img = img_as_arr(group_img)
        template = img_as_arr(indiv_img)
        # match_template gives a (H-h+1, W-w+1) correlation map; argmax -> top-left
        corr = skimage.feature.match_template(img, template, pad_input=False)
        iy, ix, _ = np.unravel_index(np.argmax(corr), corr.shape)
        return int(ix), int(iy)

    group_rel_path, indiv_rel_path = img_df.select(
        "groupImageFilePath", "individualImageFilePath"
    ).row(index=3)
    find_origin(hf_root / group_rel_path, hf_root / indiv_rel_path)
    return group_rel_path, indiv_rel_path


@app.cell
def _(Image, ImageDraw, group_rel_path, hf_root, indiv_rel_path, mo):
    def _():
        group_img = Image.open(hf_root / group_rel_path)
        indiv_img = Image.open(hf_root / indiv_rel_path)
        x, y = [2516, 275]
        content = []
        w, h = indiv_img.size
        crop_box = [x, y, x + w, y + h]
        cropped_img = group_img.crop(crop_box)
        print(group_img.size)
        draw = ImageDraw.Draw(group_img)
        draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=12)

        w, h = group_img.size
        content.append(mo.image(group_img.resize((w // 10, h // 10))))

        content.append(mo.image(indiv_img))

        content.append(mo.image(cropped_img))
        return mo.vstack(content)

    _()
    return


@app.cell
def _(anns_df, img_df, trait_df):
    final_df = (
        trait_df.join(anns_df, on=["GroupImgBasename", "BeetlePosition"], how="inner")
        .join(img_df, on=["GroupImgBasename", "BeetlePosition"], how="inner")
        .select(
            "groupImageFilePath",
            "individualImageFilePath",
            "bbox",
            "measurement_type",
            "polyline",
        )
    )
    final_df
    return (final_df,)


@app.cell
def _():
    return


@app.cell
def _(final_df):
    final_df.row(index=0)
    return


@app.cell
def _(Image, ImageDraw, final_df, hf_root, mo, pl):
    def _():
        (
            group_img_rel_path,
            ind_img_rel_path,
            (x, y, width, height),
            _,
            polyline,
        ) = final_df.filter(
            pl.col("groupImageFilePath") == "group_images/IMG_0531.png"
        ).row(index=0)
        content = []

        group_img = Image.open(hf_root / group_img_rel_path)
        ind_img = Image.open(hf_root / ind_img_rel_path)

        crop_box = [x, y, x + width, y + height]
        cropped_img = group_img.crop(crop_box)

        print(group_img.size)

        draw = ImageDraw.Draw(group_img)
        draw.rectangle((x, y, x + width, y + height), fill=(255, 0, 0), width=12)

        print(x, y, width, height)
        print(polyline)

        w, h = group_img.size
        content.append(mo.image(group_img.resize((w // 10, h // 10))))

        content.append(mo.image(ind_img))

        # w, h = cropped_img.size
        # content.append(mo.image(cropped_img.resize((w // 10, h // 10))))

        return mo.vstack(content)

    _()
    return


@app.cell
def _():
    return


@app.cell
def _(Image, ImageDraw, json, mo, root, trait_df):
    def load_all(n: int = -1):
        grouped_df = trait_df.group_by("groupImageFilePath").all()
        if n > 0:
            grouped_df = grouped_df.head(n=n)
        for im_path, coords in grouped_df.select(
            "groupImageFilePath", "coords_elytra_max_width"
        ).iter_rows():
            coords = [json.loads(c_str)[0] for c_str in coords]
            yield (im_path, coords)

    content = []
    for im_path, coords in load_all(5):
        im = Image.open(root / im_path)
        draw = ImageDraw.Draw(im)
        for co in coords:
            draw.line(co, fill=(255, 0, 0), width=12)

        w, h = im.size
        im = im.resize((w // 10, h // 10))
        content.append(mo.md(im_path))
        content.append(mo.image(im))
    mo.vstack(content)
    return


@app.cell
def _(
    Image,
    ImageDraw,
    collections,
    img_df,
    json,
    match_template,
    mo,
    np,
    pathlib,
    pl,
    root,
    trait_df,
):
    def _():
        MEASURE_JSON = pathlib.Path(
            "/fs/ess/PAS2136/Hawaii-2025/beetles_intake/BeetlePUUM/Annotations/HawaiiBeetles_Measurements.json"
        )

        # Load JSON once
        with open(MEASURE_JSON, "r") as f:
            _raw_meas = json.load(f)

        def _poly_pts(entry):
            pln = entry["polyline"]
            coords = (
                pln[0]
                if (len(pln) == 1 and isinstance(pln[0], list))
                else [c for seg in pln for c in seg]
            )
            return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

        # Index measurements: (basename, position) -> {type: [(x,y), ...]}
        meas = collections.defaultdict(lambda: collections.defaultdict(list))
        for d in _raw_meas:
            pos = d.get("BeetlePosition")
            if pos is None:  # skip scalebar-only, etc.
                continue
            base = pathlib.Path(d["toras_path"]).stem.split(".")[0]  # "IMG_0531"
            meas[(base, int(pos))][d["measurement_type"]] += _poly_pts(d)

        # Map (basename, position) -> individual image path and ID
        pos_to_indiv = (
            trait_df.select("groupImageFilePath", "BeetlePosition", "individualID")
            .unique()
            .join(
                img_df.select(
                    "individualID", "individualImageFilePath", "groupImageFilePath"
                ).rename({"groupImageFilePath": "groupImageFilePath_img"}),
                on="individualID",
                how="left",
            )
            .with_columns(
                pl.col("groupImageFilePath")
                .str.split("/")
                .list.last()
                .str.replace(".png", "")
                .alias("basename")
            )
        )

        pos_to_indiv_map = {
            (row["basename"], int(row["BeetlePosition"])): {
                "individualImageFilePath": row["individualImageFilePath"],
                "individualID": row["individualID"],
                "groupImageFilePath": row["groupImageFilePath"],
            }
            for row in pos_to_indiv.to_dicts()
        }

        def _to_gray(arr_or_im, down=1):
            im = (
                arr_or_im
                if isinstance(arr_or_im, Image.Image)
                else Image.open(arr_or_im)
            )
            if down > 1:
                im = im.resize((im.width // down, im.height // down), Image.BILINEAR)
            return np.asarray(im.convert("L"), dtype=np.float32) / 255.0

        def _register_origin(group_path, indiv_path, down=4):
            """Find where the individual image sits inside the group image.
            Returns (x0, y0) in GROUP pixels (top-left of the individual crop)."""
            G = _to_gray(group_path, down=down)
            T = _to_gray(indiv_path, down=down)
            # match_template gives a (H-h+1, W-w+1) correlation map; argmax -> top-left
            corr = match_template(G, T, pad_input=False)
            iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
            return int(ix * down), int(iy * down)

        def _shift_poly(points, origin):
            x0, y0 = origin
            return [(x - x0, y - y0) for (x, y) in points]

        def individual_annotations(basename: str, position: int, down: int = 4):
            """Return polylines in INDIVIDUAL-IMAGE pixel coords plus a quick overlay for QA."""
            key = (basename, int(position))
            if key not in meas:
                raise KeyError(f"No JSON measurements for {key}")
            if key not in pos_to_indiv_map:
                raise KeyError(f"No mapping to individual image for {key}")

            indiv_rel = pos_to_indiv_map[key]["individualImageFilePath"]
            group_rel = pos_to_indiv_map[key]["groupImageFilePath"]
            indiv_path = root / indiv_rel
            group_path = root / group_rel

            # Register exactly where the crop came from
            origin_x, origin_y = _register_origin(group_path, indiv_path, down=down)

            # Shift all polylines into the individual-image frame
            polylines_indiv = {
                t: _shift_poly(pts, (origin_x, origin_y))
                for t, pts in meas[key].items()
            }

            # Build QA overlay
            im = Image.open(indiv_path).convert("RGB")
            draw = ImageDraw.Draw(im)
            colors = {
                "elytra_max_length": (255, 0, 0),
                "elytra_max_width": (0, 0, 255),
                "basal_pronotum_width": (0, 128, 0),
                "scalebar": (0, 0, 0),
            }
            for t, pts in polylines_indiv.items():
                if len(pts) >= 2:
                    flat = [v for p in pts for v in p]
                    draw.line(flat, fill=colors.get(t, (255, 128, 0)), width=8)

            return {
                "individualImageFilePath": indiv_rel,
                "individualID": pos_to_indiv_map[key]["individualID"],
                "origin_in_group_px": (origin_x, origin_y),
                "polylines_px": polylines_indiv,
                "overlay": im,
            }

        def preview_individual_registered(
            basename: str, position: int, shrink: int = 1
        ):
            data = individual_annotations(basename, position)
            im = data["overlay"]
            if shrink > 1:
                im = im.resize((im.width // shrink, im.height // shrink))
            return mo.vstack([
                mo.md(
                    f"**{basename} / position {position}** → {data['individualImageFilePath']} (ID {data['individualID']})"
                ),
                mo.md(f"crop origin in group px: {data['origin_in_group_px']}"),
                mo.image(im),
            ])

        return preview_individual_registered("IMG_0532", 3, shrink=1)

    _()
    return


if __name__ == "__main__":
    app.run()
