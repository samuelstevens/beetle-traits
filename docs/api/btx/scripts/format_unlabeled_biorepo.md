Module btx.scripts.format_unlabeled_biorepo
===========================================
Format unlabeled CarabidImaging beetle images without template matching.

Enumerates individual beetle images in Output/cropped_images/ as-is, using the
position already encoded in each filename. No template matching, no renaming.
Writes an annotations CSV with paths and taxon metadata.

INPUTS:
- /fs/ess/PAS2136/CarabidImaging/Output/cropped_images/: Individual beetle images
- /fs/ess/PAS2136/CarabidImaging/Images/FinalImages/: Group images (for paths only)
- /fs/ess/PAS2136/CarabidImaging/allIndividuals.csv: Beetle metadata

OUTPUTS:
- annotations.csv: One row per beetle with paths and taxon metadata

USAGE:
------
  uv run python -m btx.scripts.format_unlabeled_biorepo_v2

Exclude already-labeled beetles:
  uv run python -m btx.scripts.format_unlabeled_biorepo_v2       --labeled-annotations data/biorepo-formatted/annotations.json

Functions
---------

`build_image_index(cfg: btx.scripts.format_unlabeled_biorepo.Config) ‑> dict[str, pathlib.Path]`
:   Return a mapping of group image stem -> absolute path by scanning images_dir.

`load_labeled_exclusions(labeled_fpath: pathlib.Path) ‑> set[str]`
:   Return set of individual_ids present in the labeled annotations JSON.

`load_metadata(cfg: btx.scripts.format_unlabeled_biorepo.Config) ‑> dict[tuple[str, int], dict]`
:   Return {(group_stem, order): {individual_id, taxon_id, scientific_name}}.

`main(cfg: btx.scripts.format_unlabeled_biorepo.Config) ‑> None`
:   

Classes
-------

`Config(carb_dir: pathlib.Path = PosixPath('/fs/ess/PAS2136/CarabidImaging'), dump_to: pathlib.Path = PosixPath('data/unlabeled-biorepo'), labeled_annotations: pathlib.Path | None = PosixPath('/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json'))`
:   Config(carb_dir: pathlib.Path = PosixPath('/fs/ess/PAS2136/CarabidImaging'), dump_to: pathlib.Path = PosixPath('data/unlabeled-biorepo'), labeled_annotations: pathlib.Path | None = PosixPath('/fs/scratch/PAS2136/cain429/biorepo-formatted/annotations.json'))

    ### Instance variables

    `annotations_fpath: pathlib.Path`
    :

    `carb_dir: pathlib.Path`
    :   Root of the CarabidImaging dataset.

    `cropped_dir: pathlib.Path`
    :

    `dump_to: pathlib.Path`
    :   Where to save annotations.csv.

    `images_dir: pathlib.Path`
    :

    `labeled_annotations: pathlib.Path | None`
    :   Path to labeled annotations JSON. If provided, beetles whose individual_id
        appears in the labeled data are excluded from the output CSV.

    `metadata_csv: pathlib.Path`
    :