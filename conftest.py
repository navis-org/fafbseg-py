import os

# This is to avoid some print statements in the code that are semi-random and
# hence will make doctests fails. I tried adding this environment variable to
# pytest.ini but that didn't work.
os.environ['FAFBSEG_TESTING'] = 'TRUE'

SKIP = ['conf.py']

def pytest_ignore_collect(path):
    # Do not test the Blender module - for some reason that can't be imported
    if "blender" in str(path):
        return True
    for s in SKIP:
        if str(path).endswith(s):
            return True

