import dataclasses
import pathlib

import beartype
import grain
import polars as pl


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    root: pathlib.Path = pathlib.Path("data/hawaii")


@beartype.beartype
class Dataset(grain.sources.RandomAccessDataSource):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        df = pl.read_csv(cfg.root / "trait_annotations.csv")
        breakpoint()

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict[str, object]:
        """Load image and annotations for given index."""
        ann = self.annotations[idx]

        # Load image
        img_path = os.path.join(self.img_dir, ann["image_name"])
        image = np.array(Image.open(img_path).convert("RGB"))

        return {
            "image": image,
            "keypoints": ann["keypoints"],
            "image_name": ann["image_name"],
        }
