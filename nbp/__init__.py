"""
nbp
===

Using nbp
--------------
python setup.py
import nbp
Enjoy :)

Subpackages
--------------
markov

neighbours

distance

sysmodule

unitconvert
"""

from .markov import *
from .neighbours import *
from .sysmodule import *
from .distance import *
from .unitconvert import *
from .parser import *

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time for {}: {}'.format(f.__name__, end-start))
        return result
    return wrapper
