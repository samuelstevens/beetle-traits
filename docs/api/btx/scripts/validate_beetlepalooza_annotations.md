Module btx.scripts.validate_beetlepalooza_annotations
=====================================================
Validate that elytra measurements are within the bounding box of individual beetle images.

This script reads annotations.json and checks if the measurement coordinates
(elytra_length, elytra_width) fall within the bounds of the individual image.

Functions
---------

`main(cfg: btx.scripts.validate_beetlepalooza_annotations.Config) ‑> int`
:   

`validate_annotation(annotation: dict, hf_root: pathlib.Path, logger: logging.Logger) ‑> btx.scripts.validate_beetlepalooza_annotations.ValidationResult | None`
:   Validate measurements for a single annotation.

`validate_measurement_coords(coords: dict[str, float], img_width: int, img_height: int, measurement_type: str) ‑> tuple[bool, str]`
:   Check if measurement coordinates are within image bounds.
    
    Returns (is_valid, error_message)

`visualize_invalid_measurement(result: btx.scripts.validate_beetlepalooza_annotations.ValidationResult, cfg: btx.scripts.validate_beetlepalooza_annotations.Config, logger: logging.Logger) ‑> None`
:   Draw the group image with bounding box and invalid measurements.

Classes
-------

`Config(annotations_file: pathlib.Path = PosixPath('data/beetlepalooza-formatted/annotations.json'), hf_root: pathlib.Path = PosixPath('data/beetlepalooza/individual_specimens'), resized_root: pathlib.Path = PosixPath('data/beetlepalooza/group_images_resized'), output_dir: pathlib.Path = PosixPath('data/beetlepalooza-formatted/validation-examples'), visualize_sample_rate: float = 0.1, seed: int = 42, delete_invalid: bool = True, save_invalid_separately: bool = True)`
:   Config(annotations_file: pathlib.Path = PosixPath('data/beetlepalooza-formatted/annotations.json'), hf_root: pathlib.Path = PosixPath('data/beetlepalooza/individual_specimens'), resized_root: pathlib.Path = PosixPath('data/beetlepalooza/group_images_resized'), output_dir: pathlib.Path = PosixPath('data/beetlepalooza-formatted/validation-examples'), visualize_sample_rate: float = 0.1, seed: int = 42, delete_invalid: bool = True, save_invalid_separately: bool = True)

    ### Instance variables

    `annotations_file: pathlib.Path`
    :   Path to the annotations.json file.

    `delete_invalid: bool`
    :   If True, remove invalid annotations from annotations.json.

    `hf_root: pathlib.Path`
    :   Path to individual specimens directory (to get image dimensions).

    `output_dir: pathlib.Path`
    :   Where to save visualization images.

    `resized_root: pathlib.Path`
    :   Path to group images.

    `save_invalid_separately: bool`
    :   If True, save invalid annotations to a separate file.

    `seed: int`
    :   Random seed for sampling which invalid measurements to visualize.

    `visualize_sample_rate: float`
    :   Fraction of invalid measurements to visualize (0.1 = 10%).

`ValidationResult(individual_id: str, group_img_basename: str, beetle_position: int, image_width: int, image_height: int, measurements_checked: int, measurements_valid: int, measurements_invalid: int, invalid_details: list[str] = <factory>, annotation: dict = <factory>)`
:   Results from validating one annotation.

    ### Instance variables

    `annotation: dict`
    :

    `beetle_position: int`
    :

    `group_img_basename: str`
    :

    `image_height: int`
    :

    `image_width: int`
    :

    `individual_id: str`
    :

    `invalid_details: list[str]`
    :

    `measurements_checked: int`
    :

    `measurements_invalid: int`
    :

    `measurements_valid: int`
    :