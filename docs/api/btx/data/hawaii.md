Module btx.data.hawaii
======================
Hawaii beetle dataset loader for trait prediction model training.

Goal:
Load individual beetle images with their trait annotations (elytra max length,
basal pronotum width, elytra max width) for training keypoint detection models.
Each sample should contain an individual beetle image cropped from the group
photo and the corresponding trait polylines in pixel coordinates.

Dataset Structure:
- annotations.json: Contains all beetle annotations with:
  - Individual image paths (relative to HuggingFace dataset root)
  - Origin coordinates (x, y) in the group image
  - Trait measurements as polylines in individual image coordinates
  - NCC score from template matching (confidence metric)
- Individual images: Located at HF_ROOT/individual_specimens/
- Group images: Located at HF_ROOT/group_images/ (not needed for training)

Implementation Challenges:

1. Path resolution:
   - Annotations contain absolute paths that need conversion to relative
   - Must handle both local and scratch filesystem paths
   Solution: Extract relative path components from indiv_img_rel_path field

2. Coordinate systems:
   - Polylines are already in individual image pixel coordinates
   - No transformation needed, just validation
   Solution: Directly use polyline_px coordinates from annotations

3. Data filtering:
   - Some beetles may have missing or incomplete annotations
   - NCC scores indicate template matching confidence
   Solution: Filter by NCC threshold (e.g., > 0.8) and validate all traits present

4. Memory efficiency:
   - Dataset has ~1600 beetles, loading all at once may be excessive
   - Images vary in size (typically 400-1000px per dimension)
   Solution: Use grain's lazy loading, load images on-demand in __getitem__

Unresolved Challenges:
- Handling variable image sizes for batch training (requires padding/resizing)
- Dealing with outlier beetle sizes (some are 1500+ pixels)
- Normalizing trait measurements across different beetle scales
- Variable keypoint counts: elytra_max_width has 2 points (73%) or 4 points (27%)
  when wings are spread. With 4 points, only outer segments matter (middle segment
  crosses the gap between wings). Models need fixed keypoint counts, so options:
  a) Always predict 4 points, ignore middle segment when present
  b) Only predict 2 endpoints, but this overestimates width for spread wings
  c) Predict wing spread as separate classification task, then variable points

Testing Strategy:
1. Load a few samples and visualize with trait polylines overlaid
2. Verify polyline coordinates fall within image bounds
3. Check distribution of NCC scores and image dimensions
4. Test with grain's DataLoader for batching compatibility
5. Validate against saved example images in random-examples/

Classes
-------

`Config(hf_root: pathlib.Path = PosixPath('data/hawaii'), annotations: pathlib.Path = PosixPath('data/hawaii-formatted/annotations.json'), include_polylines: bool = True, split: Literal['train', 'val'] = 'train', seed: int = 0, min_val_groups: int = 2, min_val_beetles: int = 20, n_workers: int = 4, batch_size: int = 16)`
:   Config(hf_root: pathlib.Path = PosixPath('data/hawaii'), annotations: pathlib.Path = PosixPath('data/hawaii-formatted/annotations.json'), include_polylines: bool = True, split: Literal['train', 'val'] = 'train', seed: int = 0, min_val_groups: int = 2, min_val_beetles: int = 20, n_workers: int = 4, batch_size: int = 16)

    ### Instance variables

    `annotations: pathlib.Path`
    :   Path to the annotations.json file made by running format_hawaii.py.

    `batch_size: int`
    :

    `hf_root: pathlib.Path`
    :   Path to the dataset root downloaded from HuggingFace.

    `include_polylines: bool`
    :   Whether to include polylines (lines with more than 2 points).

    `min_val_beetles: int`
    :   Minimum beetles per species in validation.

    `min_val_groups: int`
    :   Minimum group images per species in validation.

    `n_workers: int`
    :

    `seed: int`
    :   Random seed for split.

    `split: Literal['train', 'val']`
    :   Which split.

`Dataset(cfg: btx.data.hawaii.Config)`
:   Interface for datasources where storage supports efficient random access.
    
    Note that `__repr__` has to be additionally implemented to make checkpointing
    work with this source.

    ### Ancestors (in MRO)

    * grain._src.python.data_sources.RandomAccessDataSource
    * typing.Protocol
    * typing.Generic