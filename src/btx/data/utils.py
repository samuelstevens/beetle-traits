import dataclasses
import typing as tp

import beartype
import grain
import numpy as np
from jaxtyping import Float, jaxtyped
from PIL import Image


@jaxtyped(typechecker=beartype.beartype)
class Sample(tp.TypedDict):
    img_fpath: str

    points_px: Float[np.ndarray, "2 2 2"]
    """{width, length} x two points x {x, y}."""

    # Metadata
    beetle_id: str
    beetle_position: int
    group_img_basename: str


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class DecodeRGB(grain.transforms.Map):
    def map(self, sample: Sample) -> Sample:
        # Heavy I/O lives in a transform so workers can parallelize it
        with Image.open(sample["img_fpath"]) as im:
            sample["img"] = im.convert("RGB")
        return sample


@dataclasses.dataclass(frozen=True)
class Resize(grain.transforms.Map):
    size: int = 256

    def map(self, sample: dict[str, object]) -> dict[str, object]:
        img = sample["img"]
        orig_h, orig_w = img.size

        img = np.array(img.resize((self.size, self.size)))
        sample["img"] = img.astype(np.float32) / 255.0

        # Rescale the measurements according to the new size
        scale_x = self.size / orig_w
        scale_y = self.size / orig_h

        points = sample["points_px"].copy()
        points[:, :, 0] *= scale_x
        points[:, :, 1] *= scale_y
        sample["tgt"] = points

        sample["scale"] = np.array([scale_x, scale_y])

        return sample
