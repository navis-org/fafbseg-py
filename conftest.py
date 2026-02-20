import os
from pathlib import Path

# This is to avoid some print statements in the code that are semi-random and
# hence will make doctests fails. I tried adding this environment variable to
# pytest.ini but that didn't work.
os.environ["FAFBSEG_TESTING"] = "TRUE"

SKIP = ["conf.py"]


def pytest_ignore_collect(collection_path: Path, config):
    """Return True to prevent considering this path for collection.
    This hook is consulted for all files and directories prior to calling
    more specific hooks.
    """
    path = str(collection_path)
    if "blender" in str(path):
        return True

    for s in SKIP:
        if str(path).endswith(s):
            return True
