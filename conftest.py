
SKIP = ['conf.py']

def pytest_ignore_collect(path):
    # Do not test the Blender module - for some reason that can't be imported
    if "blender" in str(path):
        return True
    for s in SKIP:
        if str(path).endswith(s):
            return True

