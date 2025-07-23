# src/btx/scripts/download_hawaii.py
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "requests",
#     "tyro",
# ]
# ///
import pathlib

import huggingface_hub as hfhub
import tyro


def main(dump_to: pathlib.Path = pathlib.Path("data/hawaii")):
    hfhub.snapshot_download(
        repo_id="imageomics/Hawaii-beetles",
        repo_type="dataset",
        local_dir=str(dump_to),
        # allow_patterns=["individual_specimens/*.png", "*.csv", "README.md"],
        allow_patterns=["*.png", "*.csv", "README.md"],
        max_workers=8,
    )


if __name__ == "__main__":
    tyro.cli(main)
