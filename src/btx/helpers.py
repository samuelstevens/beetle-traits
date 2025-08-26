# src/btx/helpers.py
import collections.abc
import logging
import pathlib
import re
import subprocess
import time

import beartype


@beartype.beartype
def fssafe(s: str) -> str:
    """Convert a string to be filesystem-safe by replacing special characters.

    This is particularly useful for checkpoint names that contain characters like
    'hf-hub:timm/ViT-L-16-SigLIP2-256' which need to be converted to something like
    'hf-hub_timm_ViT-L-16-SigLIP2-256'.

    Args:
        s: String to make filesystem-safe.

    Returns:
        Filesystem-safe version of the string.
    """
    # Replace common problematic characters with underscores
    replacements = {
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
        " ": "_",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    # Remove any remaining non-alphanumeric characters except - _ .
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


@beartype.beartype
class progress:
    def __init__(self, it, *, every: int = 10, desc: str = "progress", total: int = 0):
        """Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish.

        Args:
            it: Iterable to wrap.
            every: How many iterations between logging progress.
            desc: What to name the logger.
            total: If non-zero, how long the iterable is.
        """
        self.it = it
        self.every = max(1, every)
        self.logger = logging.getLogger(desc)
        self.total = total

    def __iter__(self):
        start = time.time()

        try:
            total = len(self)
        except TypeError:
            total = None

        for i, obj in enumerate(self.it):
            yield obj

            if (i + 1) % self.every == 0:
                now = time.time()
                duration_s = now - start
                per_min = (i + 1) / (duration_s / 60)

                if total is not None:
                    pred_min = (total - (i + 1)) / per_min
                    self.logger.info(
                        "%d/%d (%.1f%%) | %.1f it/m (expected finish in %.1fm)",
                        i + 1,
                        total,
                        (i + 1) / total * 100,
                        per_min,
                        pred_min,
                    )
                else:
                    self.logger.info("%d/? | %.1f it/m", i + 1, per_min)

    def __len__(self) -> int:
        if self.total > 0:
            return self.total

        # Will throw exception.
        return len(self.it)


@beartype.beartype
class batched_idx:
    """Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size.

    Args:
        total_size: total number of examples
        batch_size: maximum distance between the generated indices.

    Returns:
        A generator of (int, int) tuples that can slice up a list or a tensor.
    """

    def __init__(self, total_size: int, batch_size: int):
        self.total_size = total_size
        self.batch_size = batch_size

    def __iter__(self) -> collections.abc.Iterator[tuple[int, int]]:
        """Yield (start, end) index pairs for batching."""
        for start in range(0, self.total_size, self.batch_size):
            stop = min(start + self.batch_size, self.total_size)
            yield start, stop

    def __len__(self) -> int:
        """Return the number of batches."""
        return (self.total_size + self.batch_size - 1) // self.batch_size


@beartype.beartype
def current_git_commit() -> str | None:
    """
    Best-effort short SHA of the repo containing *this* file.

    Returns `None` when
    * `git` executable is missing,
    * we’re not inside a git repo (e.g. installed wheel),
    * or any git call errors out.
    """
    try:
        # Walk up until we either hit a .git dir or the FS root
        here = pathlib.Path(__file__).resolve()
        for parent in (here, *here.parents):
            if (parent / ".git").exists():
                break
        else:  # no .git found
            return None

        result = subprocess.run(
            ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


@beartype.beartype
def get_slurm_max_array_size() -> int:
    """
    Get the MaxArraySize configuration from the current Slurm cluster.

    Returns:
        int: The maximum array size allowed on the cluster. Returns 1000 as fallback if unable to determine.
    """
    logger = logging.getLogger("helpers.slurm")
    try:
        # Run scontrol command to get config information
        result = subprocess.run(
            ["scontrol", "show", "config"], capture_output=True, text=True, check=True
        )

        # Search for MaxArraySize in the output
        match = re.search(r"MaxArraySize\s*=\s*(\d+)", result.stdout)
        if match:
            max_array_size = int(match.group(1))
            logger.info("Detected MaxArraySize = %d", max_array_size)
            return max_array_size
        else:
            logger.warning(
                "Could not find MaxArraySize in scontrol output, using default of 1000"
            )
            return 1000

    except subprocess.SubprocessError as e:
        logger.error("Error running scontrol: %s", e)
        return 1000  # Safe default
    except ValueError as e:
        logger.error("Error parsing MaxArraySize: %s", e)
        return 1000  # Safe default
    except FileNotFoundError:
        logger.warning(
            "scontrol command not found. Assuming not in Slurm environment. Returning default MaxArraySize=1000."
        )
        return 1000


@beartype.beartype
def get_slurm_max_submit_jobs() -> int:
    """
    Get the MaxSubmitJobs limit from the current user's QOS.

    Returns:
        int: The maximum number of jobs that can be submitted at once. Returns 1000 as fallback.
    """
    logger = logging.getLogger("helpers.slurm")
    try:
        # First, try to get the QOS from a recent job
        result = subprocess.run(
            ["scontrol", "show", "job", "-o"],
            capture_output=True,
            text=True,
            check=False,
        )

        qos_name = None
        if result.returncode == 0 and result.stdout:
            # Extract QOS from job info
            match = re.search(r"QOS=(\S+)", result.stdout)
            if match:
                qos_name = match.group(1)

        if not qos_name:
            # If no jobs, try to get default QOS from association
            # This is less reliable but better than nothing
            logger.warning("No active jobs to determine QOS, using default of 1000")
            return 1000

        # Get the MaxSubmitJobs for this QOS
        result = subprocess.run(
            ["sacctmgr", "show", "qos", qos_name, "format=maxsubmitjobs", "-n", "-P"],
            capture_output=True,
            text=True,
            check=True,
        )

        max_submit = result.stdout.strip()
        if max_submit and max_submit.isdigit():
            limit = int(max_submit)
            logger.info("Detected MaxSubmitJobs = %d for QOS %s", limit, qos_name)
            return limit
        else:
            logger.warning("Could not parse MaxSubmitJobs, using default of 1000")
            return 1000

    except subprocess.SubprocessError as e:
        logger.error("Error getting MaxSubmitJobs: %s", e)
        return 1000
    except (ValueError, FileNotFoundError) as e:
        logger.error("Error: %s", e)
        return 1000


@beartype.beartype
def get_slurm_job_count() -> int:
    """
    Get the current number of jobs in the queue for the current user.

    Uses squeue's -r flag to properly count job array elements individually.
    For example, a job array 12345_[0-99] will be counted as 100 jobs.
    """
    try:
        # Use -r to display each array element on its own line
        result = subprocess.run(
            ["squeue", "--me", "-h", "-r"], capture_output=True, text=True, check=True
        )

        # Count non-empty lines
        lines = result.stdout.strip().split("\n")
        return len([line for line in lines if line.strip()])

    except (subprocess.SubprocessError, FileNotFoundError):
        # If we can't check, assume no jobs
        return 0
