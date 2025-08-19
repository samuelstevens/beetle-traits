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


def main(dump_to: pathlib.Path = pathlib.Path("data/beetlepalooza")):
    hfhub.snapshot_download(
        repo_id="imageomics/2018-NEON-beetles",
        repo_type="dataset",
        local_dir=str(dump_to),
        allow_patterns=[
            "individual_specimens/**/*.png",
            "*.csv",
            "README.md",
        ],
        revision="refs/pr/25",
        max_workers=1,
    )


if __name__ == "__main__":
    tyro.cli(main)
