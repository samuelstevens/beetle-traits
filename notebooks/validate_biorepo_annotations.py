import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    return Image, Path, json, mo, np, plt


@app.cell
def _(Path, json):

    # Find project root - go up from notebooks to beetle-traits
    notebook_path = Path(__file__).resolve()
    # When running in marimo, __file__ is in notebooks/__marimo__/
    # So we need to go up 2 levels to get to beetle-traits
    if "__marimo__" in str(notebook_path):
        project_root = notebook_path.parent.parent.parent
    else:
        project_root = notebook_path.parent.parent

    print(f"Project root: {project_root}")

    annotations_path = project_root / "data" / "biorepo-formatted" / "annotations.json"
    biorepo_dir = project_root / "data" / "biorepo"

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} beetle annotations")
    return annotations, biorepo_dir


@app.cell
def _(annotations):
    # Get unique group images and build lookup
    unique_images = sorted(set(a["group_img"] for a in annotations))
    print(f"Found {len(unique_images)} unique group images")

    # Build a lookup: group_img -> list of beetle annotations
    annotations_by_image = {}
    for a in annotations:
        img_key = a["group_img"]
        if img_key not in annotations_by_image:
            annotations_by_image[img_key] = []
        annotations_by_image[img_key].append(a)

    # Sort beetles by position within each image
    for img_key in annotations_by_image:
        annotations_by_image[img_key].sort(key=lambda x: x["beetle_position"])
    return annotations_by_image, unique_images


@app.cell
def _(mo, unique_images):
    # Create dropdown for selecting group image
    image_dropdown = mo.ui.dropdown(
        options={img: img for img in unique_images},
        value=unique_images[0] if unique_images else None,
        label="Select Group Image",
    )
    image_dropdown
    return (image_dropdown,)


@app.cell
def _(annotations_by_image, image_dropdown, mo):
    # Create dropdown for selecting beetle based on selected image
    selected_image = image_dropdown.value
    beetles_in_image = annotations_by_image.get(selected_image, [])
    beetle_options = {
        f"Beetle {b['beetle_position']}": b["beetle_position"] for b in beetles_in_image
    }

    beetle_dropdown = mo.ui.dropdown(
        options=beetle_options,
        value=list(beetle_options.keys())[0] if beetle_options else None,
        label="Select Beetle",
    )
    beetle_dropdown
    return (beetle_dropdown,)


@app.cell
def _(annotations_by_image, beetle_dropdown, image_dropdown):
    # Get the selected annotation
    selected_img = image_dropdown.value
    selected_beetle_num = beetle_dropdown.value

    selected_annotation = None
    for entry in annotations_by_image.get(selected_img, []):
        if entry["beetle_position"] == selected_beetle_num:
            selected_annotation = entry
            break

    if selected_annotation:
        print(f"Selected: {selected_img}, Beetle {selected_beetle_num}")
        print(f"Scientific name: {selected_annotation.get('scientific_name', 'N/A')}")
        print(f"Taxon ID: {selected_annotation.get('taxon_id', 'N/A')}")
        print(
            f"Offset: ({selected_annotation['offset_x']:.1f}, {selected_annotation['offset_y']:.1f})"
        )
    return (selected_annotation,)


app._unparsable_cell(
    r"""
    if selected_annotation is None:
        # No annotation selected; skip visualization to avoid errors.
        return
    # Load images
    group_img_path = biorepo_dir / "Images" / selected_annotation["group_img"]
    individual_img_path = biorepo_dir / selected_annotation["rel_individual_img_path"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left panel: Group image with bounding box showing beetle location
    if group_img_path.exists():
        group_img = np.array(Image.open(group_img_path))
        axes[0].imshow(group_img)

        # Draw bounding box around beetle region
        offset_x = selected_annotation["offset_x"]
        offset_y = selected_annotation["offset_y"]

        if individual_img_path.exists():
            indiv_img_pil = Image.open(individual_img_path)
            w, h = indiv_img_pil.size
            rect = plt.Rectangle(
                (offset_x, offset_y),
                w,
                h,
                fill=False,
                edgecolor="cyan",
                linewidth=2,
                linestyle="--",
            )
            axes[0].add_patch(rect)

        axes[0].set_title(
            f"Group Image: {selected_annotation['group_img']}\n"
            f"Beetle {selected_annotation['beetle_position']} location",
            fontsize=12,
        )
        axes[0].axis("off")
    else:
        axes[0].text(
            0.5,
            0.5,
            f"Group image not found:\n{group_img_path}",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].axis("off")

    # Right panel: Individual beetle image with annotations
    if individual_img_path.exists():
        indiv_img = np.array(Image.open(individual_img_path))
        axes[1].imshow(indiv_img)

        # Draw measurements on individual image (coordinates are now relative to individual image)
        colors = {
            "elytra_length": "red",
            "elytra_width": "blue",
            "pronotum_width": "green",
            "scalebar": "yellow",
        }
        labels_added = set()

        for measurement in selected_annotation["measurements"]:
            mtype = measurement["measurement_type"]
            polyline = measurement["polyline"]

            if polyline:
                xs = [pt[0] for pt in polyline]
                ys = [pt[1] for pt in polyline]

                label = mtype if mtype not in labels_added else None
                labels_added.add(mtype)

                axes[1].plot(
                    xs,
                    ys,
                    color=colors.get(mtype, "white"),
                    marker="o",
                    markersize=6,
                    linewidth=2,
                    label=label,
                    markeredgecolor="white",
                )

        axes[1].legend(loc="upper right")
        axes[1].set_title(
            f"Individual: {individual_img_path.name}\n"
            f"{selected_annotation.get('scientific_name', 'N/A')}",
            fontsize=12,
        )
        axes[1].axis("off")
    else:
        axes[1].text(
            0.5,
            0.5,
            f"Individual image not found:\n{individual_img_path}",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].axis("off")

    plt.tight_layout()
    mo.output.replace(mo.mpl.interactive(fig))
    """,
    name="_",
)


@app.cell
def _(mo, selected_annotation):
    # Show annotation details as JSON
    if selected_annotation:
        mo.md(f"""
        ## Annotation Details

        ```json
        {selected_annotation}
        ```
        """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Group Image Overview
    View all beetles and their annotations on a single group image
    """)
    return


@app.cell
def _(mo, unique_images):
    # Separate dropdown for group image overview
    group_overview_dropdown = mo.ui.dropdown(
        options={img: img for img in unique_images},
        value=unique_images[0] if unique_images else None,
        label="Select Group Image for Overview",
    )
    group_overview_dropdown
    return (group_overview_dropdown,)


@app.cell
def _(
    Image,
    annotations_by_image,
    biorepo_dir,
    group_overview_dropdown,
    np,
    plt,
):
    from matplotlib.lines import Line2D

    # Draw all beetles and annotations on the group image
    ov_selected_img = group_overview_dropdown.value
    ov_beetles = annotations_by_image.get(ov_selected_img, [])

    ov_img_path = biorepo_dir / "Images" / ov_selected_img

    ov_img_pil = Image.open(ov_img_path)
    ov_orig_w, ov_orig_h = ov_img_pil.size

    # Downsample for display (max 1500px on longest side)
    ov_max_size = 1500
    ov_scale = min(ov_max_size / ov_orig_w, ov_max_size / ov_orig_h, 1.0)
    ov_new_w = int(ov_orig_w * ov_scale)
    ov_new_h = int(ov_orig_h * ov_scale)
    ov_img_resized = ov_img_pil.resize((ov_new_w, ov_new_h), Image.Resampling.LANCZOS)
    ov_img = np.array(ov_img_resized)

    # Create figure - size based on image aspect ratio
    ov_img_h, ov_img_w = ov_img.shape[:2]
    ov_fig_width = 12
    ov_fig_height = ov_fig_width * (ov_img_h / ov_img_w)
    ov_fig, ov_ax = plt.subplots(1, 1, figsize=(ov_fig_width, ov_fig_height), dpi=100)

    ov_ax.imshow(ov_img)

    # Colors for different measurement types
    ov_colors = {
        "elytra_length": "red",
        "elytra_width": "blue",
        "pronotum_width": "green",
        "scalebar": "yellow",
    }

    # Draw each beetle's annotations (scaled to resized image)
    for ov_ann in ov_beetles:
        ov_beetle_pos = ov_ann["beetle_position"]
        ov_offset_x = ov_ann["offset_x"] * ov_scale
        ov_offset_y = ov_ann["offset_y"] * ov_scale

        # Get individual image dimensions for bounding box
        ov_indiv_path = biorepo_dir / ov_ann["rel_individual_img_path"]
        if ov_indiv_path.exists():
            ov_indiv_img = Image.open(ov_indiv_path)
            ov_w, ov_h = ov_indiv_img.size
            ov_w_scaled = ov_w * ov_scale
            ov_h_scaled = ov_h * ov_scale

            # Draw bounding box
            ov_rect = plt.Rectangle(
                (ov_offset_x, ov_offset_y),
                ov_w_scaled,
                ov_h_scaled,
                fill=False,
                edgecolor="cyan",
                linewidth=2,
                linestyle="--",
            )
            ov_ax.add_patch(ov_rect)

            # Add beetle number label
            ov_ax.text(
                ov_offset_x + ov_w_scaled / 2,
                ov_offset_y - 5,
                f"#{ov_beetle_pos}",
                color="cyan",
                fontsize=10,
                ha="center",
                va="bottom",
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            )

        # Draw measurements (convert from individual coords back to group coords, then scale)
        for ov_meas in ov_ann["measurements"]:
            ov_mtype = ov_meas["measurement_type"]
            ov_polyline = ov_meas["polyline"]

            if (
                ov_polyline and ov_mtype != "scalebar"
            ):  # Skip scalebar as it's already in group coords
                # Convert individual image coords to group image coords, then scale
                ov_xs = [(pt[0] + ov_ann["offset_x"]) * ov_scale for pt in ov_polyline]
                ov_ys = [(pt[1] + ov_ann["offset_y"]) * ov_scale for pt in ov_polyline]

                ov_ax.plot(
                    ov_xs,
                    ov_ys,
                    color=ov_colors.get(ov_mtype, "white"),
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    markeredgecolor="white",
                )

    # Add legend
    ov_legend_elements = [
        Line2D([0], [0], color="red", linewidth=2, marker="o", label="elytra_length"),
        Line2D([0], [0], color="blue", linewidth=2, marker="o", label="elytra_width"),
        Line2D(
            [0], [0], color="green", linewidth=2, marker="o", label="pronotum_width"
        ),
        Line2D(
            [0], [0], color="cyan", linewidth=2, linestyle="--", label="beetle bbox"
        ),
    ]
    ov_ax.legend(handles=ov_legend_elements, loc="upper right", fontsize=10)

    ov_ax.set_title(
        f"Group Image: {ov_selected_img}\n{len(ov_beetles)} beetles", fontsize=14
    )
    ov_ax.axis("off")

    plt.tight_layout()
    ov_fig
    return


if __name__ == "__main__":
    app.run()
