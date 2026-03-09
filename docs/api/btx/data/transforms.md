Module btx.data.transforms
==========================

Functions
---------

`apply_affine_to_points(affine_33: jaxtyping.Float[ndarray, '3 3'], points_l22: jaxtyping.Float[ndarray, 'lines 2 2']) ‑> jaxtyping.Float[ndarray, 'lines 2 2']`
:   

`get_crop_resize_affine(x0: float, y0: float, crop_w: float, crop_h: float, *, size: int) ‑> jaxtyping.Float[ndarray, '3 3']`
:   

`get_hflip_affine(*, size: int) ‑> jaxtyping.Float[ndarray, '3 3']`
:   

`get_identity_affine() ‑> jaxtyping.Float[ndarray, '3 3']`
:   

`get_rotation_affine(angle_deg: float, *, size: int) ‑> jaxtyping.Float[ndarray, '3 3']`
:   Affine for counterclockwise rotation by angle_deg around the image center.

`get_vflip_affine(*, size: int) ‑> jaxtyping.Float[ndarray, '3 3']`
:   

`is_in_bounds(points_l22: jaxtyping.Float[ndarray, 'lines 2 2'], *, size: int) ‑> jaxtyping.Bool[ndarray, 'lines 2']`
:   

`make_transforms(cfg: btx.data.transforms.AugmentConfig, *, is_train: bool) ‑> list[grain._src.core.transforms.MapTransform | grain._src.core.transforms.RandomMapTransform]`
:   Build the Grain transform list for train or eval.
    
    Args:
        cfg: Augmentation settings, target sizing, and optional normalization controls.
        is_train: Whether to build the train pipeline. If false, build the eval pipeline.
    
    Returns:
        Ordered `grain.transforms.Map`/`RandomMap` transforms to apply to each sample.
    
    The pipeline always starts with `DecodeRGB` and `InitAugState` and always applies `FinalizeTargets` before optional normalization. If `is_train` and `cfg.go` are both true, the pipeline includes stochastic spatial/color augmentation (`RandomResizedCrop`, `RandomFlip`, `RandomRotation`, `ColorJitter`). Otherwise it uses deterministic `Resize`.

Classes
-------

`AugmentConfig(go: bool = True, size: int = 256, crop: bool = True, crop_scale_min: float = 0.5, crop_scale_max: float = 1.0, crop_ratio_min: float = 0.75, crop_ratio_max: float = 1.333, hflip_prob: float = 0.5, vflip_prob: float = 0.5, rotation_prob: float = 0.75, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2, hue: float = 0.1, color_jitter_prob: float = 1.0, normalize: bool = True, oob_policy: Literal['mask_any_oob', 'mask_all_oob', 'supervise_oob'] = 'supervise_oob', min_px_per_cm: float = 1e-06)`
:   Configuration for train-time augmentation and cm-metric masking behavior.

    ### Instance variables

    `brightness: float`
    :   Color jitter brightness strength.

    `color_jitter_prob: float`
    :   Probability of applying color jitter when any jitter strength is nonzero.

    `contrast: float`
    :   Color jitter contrast strength.

    `crop: bool`
    :   Whether to use RandomResizedCrop (True) or plain Resize (False) during training. Set to False for tightly-cropped datasets where random cropping would lose too much content.

    `crop_ratio_max: float`
    :   Maximum random crop aspect ratio for RandomResizedCrop.

    `crop_ratio_min: float`
    :   Minimum random crop aspect ratio for RandomResizedCrop.

    `crop_scale_max: float`
    :   Maximum random crop area scale for RandomResizedCrop.

    `crop_scale_min: float`
    :   Minimum random crop area scale for RandomResizedCrop.

    `go: bool`
    :   Whether to enable the augmentation pipeline.

    `hflip_prob: float`
    :   Probability of applying horizontal flip.

    `hue: float`
    :   Color jitter hue strength.

    `min_px_per_cm: float`
    :   Minimum valid scalebar length in pixels for cm metrics.

    `normalize: bool`
    :   Whether to apply ImageNet normalization at the end of the transform pipeline.

    `oob_policy: Literal['mask_any_oob', 'mask_all_oob', 'supervise_oob']`
    :   Out-of-bounds supervision policy.

    `rotation_prob: float`
    :   Probability of applying a random rotation (uniform 0-360 degrees).

    `saturation: float`
    :   Color jitter saturation strength.

    `size: int`
    :   Output image side length in pixels. Fixed to 256 for this experiment.

    `vflip_prob: float`
    :   Probability of applying vertical flip.

`ColorJitter(cfg: btx.data.transforms.AugmentConfig)`
:   ColorJitter(cfg: btx.data.transforms.AugmentConfig)

    ### Ancestors (in MRO)

    * grain._src.core.transforms.RandomMapTransform
    * abc.ABC

    ### Instance variables

    `cfg: btx.data.transforms.AugmentConfig`
    :

    ### Methods

    `random_map(self, element: object, rng: numpy.random._generator.Generator) ‑> object`
    :

`DecodeRGB()`
:   DecodeRGB()

    ### Ancestors (in MRO)

    * grain._src.core.transforms.MapTransform
    * abc.ABC

    ### Methods

    `map(self, element: object) ‑> object`
    :   Maps a single element.

`FinalizeTargets(cfg: btx.data.transforms.AugmentConfig)`
:   FinalizeTargets(cfg: btx.data.transforms.AugmentConfig)

    ### Ancestors (in MRO)

    * grain._src.core.transforms.MapTransform
    * abc.ABC

    ### Instance variables

    `cfg: btx.data.transforms.AugmentConfig`
    :

    ### Methods

    `map(self, element: object) ‑> object`
    :   Maps a single element.

`InitAugState(size: int = 256, min_px_per_cm: float = 1e-06)`
:   InitAugState(size: int = 256, min_px_per_cm: float = 1e-06)

    ### Ancestors (in MRO)

    * grain._src.core.transforms.MapTransform
    * abc.ABC

    ### Instance variables

    `min_px_per_cm: float`
    :

    `size: int`
    :

    ### Methods

    `map(self, element: object) ‑> object`
    :   Maps a single element.

`Normalize(mean: tuple[float, float, float] = (0.485, 0.456, 0.406), std: tuple[float, float, float] = (0.229, 0.224, 0.225))`
:   Normalize(mean: tuple[float, float, float] = (0.485, 0.456, 0.406), std: tuple[float, float, float] = (0.229, 0.224, 0.225))

    ### Ancestors (in MRO)

    * grain._src.core.transforms.MapTransform
    * abc.ABC

    ### Instance variables

    `mean: tuple[float, float, float]`
    :

    `std: tuple[float, float, float]`
    :

    ### Methods

    `map(self, element: object) ‑> object`
    :   Maps a single element.

`RandomFlip(cfg: btx.data.transforms.AugmentConfig)`
:   RandomFlip(cfg: btx.data.transforms.AugmentConfig)

    ### Ancestors (in MRO)

    * grain._src.core.transforms.RandomMapTransform
    * abc.ABC

    ### Instance variables

    `cfg: btx.data.transforms.AugmentConfig`
    :

    ### Methods

    `random_map(self, element: object, rng: numpy.random._generator.Generator) ‑> object`
    :

`RandomResizedCrop(cfg: btx.data.transforms.AugmentConfig)`
:   RandomResizedCrop(cfg: btx.data.transforms.AugmentConfig, _max_attempts: int = 10)

    ### Ancestors (in MRO)

    * grain._src.core.transforms.RandomMapTransform
    * abc.ABC

    ### Instance variables

    `cfg: btx.data.transforms.AugmentConfig`
    :

    ### Methods

    `random_map(self, element: object, rng: numpy.random._generator.Generator) ‑> object`
    :

`RandomRotation(cfg: btx.data.transforms.AugmentConfig)`
:   Uniform random rotation from 0-360 degrees.

    ### Ancestors (in MRO)

    * grain._src.core.transforms.RandomMapTransform
    * abc.ABC

    ### Instance variables

    `cfg: btx.data.transforms.AugmentConfig`
    :

    ### Methods

    `random_map(self, element: object, rng: numpy.random._generator.Generator) ‑> object`
    :

`Resize(size: int = 256)`
:   Resize(size: int = 256)

    ### Ancestors (in MRO)

    * grain._src.core.transforms.MapTransform
    * abc.ABC

    ### Instance variables

    `size: int`
    :

    ### Methods

    `map(self, element: object) ‑> object`
    :   Maps a single element.