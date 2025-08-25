# tests/conftest.py
import os


def pytest_addoption(parser):
    parser.addoption(
        "--shards",
        action="store",
        default=None,
        help="Root directory with activation shards",
    )
    parser.addoption(
        "--dinov3-pt-ckpts",
        action="store",
        default=None,
        help="Path to DINOv3 PyTorch checkpoint directory.",
    )
    parser.addoption(
        "--dinov3-jax-ckpts",
        action="store",
        default=None,
        help="Path to DINOv3 Jax checkpoint directory.",
    )


def pytest_generate_tests(metafunc):
    """
    If a test asks for 'jax_path', parametrize it over files found in --dinov3-jax-ckpts.
    If the option isn't supplied, the test will be collected with zero params (i.e., skipped).
    """
    if "jax_path" in metafunc.fixturenames:
        root = metafunc.config.getoption("--dinov3-jax-ckpts")
        if not root:
            fnames = []
        else:
            fnames = os.listdir(root)

        paths = sorted([
            os.path.join(root, fname) for fname in fnames if fname.endswith(".eqx")
        ])

        # produce stable, human-readable IDs
        ids = [os.path.basename(p) for p in paths]

        # If nothing to test, emit zero params (no tests will run for this function)
        metafunc.parametrize("jax_path", paths, ids=ids)

    if "vit_paths" in metafunc.fixturenames:
        pt_root = metafunc.config.getoption("--dinov3-pt-ckpts")
        jax_root = metafunc.config.getoption("--dinov3-jax-ckpts")
        if not pt_root or not jax_root:
            metafunc.parametrize("vits", [], ids=[])
            return

        paths, ids = [], []
        for pt_fname in os.listdir(pt_root):
            for jax_fname in os.listdir(jax_root):
                stem, _ = os.path.splitext(jax_fname)
                if stem in pt_fname:
                    paths.append((
                        (os.path.join(pt_root, pt_fname), stem),
                        os.path.join(jax_root, jax_fname),
                    ))
                    ids.append(stem)

        # If nothing to test, emit zero params (no tests will run for this function)
        metafunc.parametrize("vit_paths", paths, ids=ids)
