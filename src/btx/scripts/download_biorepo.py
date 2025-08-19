# src/btx/scripts/download_biorepo.py
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


def main(dump_to: pathlib.Path = pathlib.Path("data/biorepo")):
    print("uv run --with huggingface_hub hf auth login")
    hfhub.snapshot_download(
        repo_id="imageomics/sentinel-beetles",
        repo_type="dataset",
        local_dir=str(dump_to),
        allow_patterns=[
            "data/*.parquet",
            "*.csv",
            "README.md",
        ],
        max_workers=8,
    )


if __name__ == "__main__":
    tyro.cli(main)
