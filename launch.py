import tyro.extras

import inference
import rank
import train

if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "train": train.main,
        "inference": inference.main,
        "rank": rank.main,
    })
