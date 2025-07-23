import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import marimo as mo
    import pathlib
    from PIL import Image, ImageDraw
    import json
    return Image, ImageDraw, json, mo, pathlib, pl


@app.cell
def _(pathlib):
    root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/datasets/hawaii-beetles")
    return (root,)


@app.cell
def _(trait_df):
    trait_df
    return


@app.cell
def _(pl, root):
    img_df = pl.read_csv(root / "images_metadata.csv")
    img_df
    return


@app.cell
def _():
    return


@app.cell
def _(Image, ImageDraw, json, mo, pl, root):
    trait_df = pl.read_csv(root / "trait_annotations.csv")


    def load_all(n: int = -1):
        grouped_df = trait_df.group_by('groupImageFilePath').all()
        if n > 0:
            grouped_df = grouped_df.head(n=n)
        for im_path, coords in grouped_df.select('groupImageFilePath', 'coords_elytra_max_width').iter_rows():
            coords = [json.loads(c_str)[0] for c_str in coords]
            yield(im_path, coords)


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
    return (trait_df,)


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
