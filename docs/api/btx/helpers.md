Module btx.helpers
==================

Functions
---------

`current_git_commit() ‑> str | None`
:   Best-effort short SHA of the repo containing *this* file.
    
    Returns `None` when
    * `git` executable is missing,
    * we’re not inside a git repo (e.g. installed wheel),
    * or any git call errors out.

`fssafe(s: str) ‑> str`
:   Convert a string to be filesystem-safe by replacing special characters.
    
    This is particularly useful for checkpoint names that contain characters like
    'hf-hub:timm/ViT-L-16-SigLIP2-256' which need to be converted to something like
    'hf-hub_timm_ViT-L-16-SigLIP2-256'.
    
    Args:
        s: String to make filesystem-safe.
    
    Returns:
        Filesystem-safe version of the string.

`get_slurm_job_count() ‑> int`
:   Get the current number of jobs in the queue for the current user.
    
    Uses squeue's -r flag to properly count job array elements individually.
    For example, a job array 12345_[0-99] will be counted as 100 jobs.

`get_slurm_max_array_size() ‑> int`
:   Get the MaxArraySize configuration from the current Slurm cluster.
    
    Returns:
        int: The maximum array size allowed on the cluster. Returns 1000 as fallback if unable to determine.

`get_slurm_max_submit_jobs() ‑> int`
:   Get the MaxSubmitJobs limit from the current user's QOS.
    
    Returns:
        int: The maximum number of jobs that can be submitted at once. Returns 1000 as fallback.

Classes
-------

`batched_idx(total_size: int, batch_size: int)`
:   Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size.
    
    Args:
        total_size: total number of examples
        batch_size: maximum distance between the generated indices.
    
    Returns:
        A generator of (int, int) tuples that can slice up a list or a tensor.

`progress(it, *, every: int = 10, desc: str = 'progress', total: int = 0)`
:   Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish.
    
    Args:
        it: Iterable to wrap.
        every: How many iterations between logging progress.
        desc: What to name the logger.
        total: If non-zero, how long the iterable is.