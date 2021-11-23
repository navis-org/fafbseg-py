#    A collection of tools to interface with manually traced and autosegmented
#    data in FAFB.
#
#    Copyright (C) 2019 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
"""Collection of utility functions."""

import requests

from functools import wraps
from urllib.parse import urlparse, urlencode

use_pbars = True


def never_cache(function):
    """Decorate to prevent caching of server responses."""
    @wraps(function)
    def wrapper(*args, **kwargs):
        # Find CATMAID instances
        instances = [v for k, v in kwargs.items() if '_instance' in k]

        # Keep track of old caching settings
        old_values = [i.caching for i in instances]
        # Set caching to False
        for rm in instances:
            rm.caching = False
        try:
            # Execute function
            res = function(*args, **kwargs)
        except BaseException:
            raise
        finally:
            # Set caching to old value
            for rm, old in zip(instances, old_values):
                rm.caching = old
        # Return result
        return res
    return wrapper


def is_url(x):
    """Check if URL is valid."""
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc, result.path])
    except BaseException:
        return False


def make_url(*args, **GET):
    """Generate URL.

    Parameters
    ----------
    *args
                Will be turned into the URL. For example::

                    >>> make_url('http://my-server.com', 'skeleton', 'list')
                    'http://my-server.com/skeleton/list'

    **GET
                Keyword arguments are assumed to be GET request queries
                and will be encoded in the url. For example::

                    >>> make_url('http://my-server.com', 'skeleton', node_gt: 100)
                    'http://my-server.com/skeleton?node_gt=100'

    Returns
    -------
    url :       str

    """
    # Generate the URL
    url = args[0]
    for arg in args[1:]:
        arg_str = str(arg)
        joiner = '' if url.endswith('/') else '/'
        relative = arg_str[1:] if arg_str.startswith('/') else arg_str
        url = requests.compat.urljoin(url + joiner, relative)
    if GET:
        url += '?{}'.format(urlencode(GET))
    return url
