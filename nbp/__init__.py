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
distance

markov

neighbours

parameters

parser

sysmodule

unitconvert
"""

from .distance import *
from .markov import *
from .neighbours import *
from .parameters import *
from .parser import *
from .sysmodule import *
from .unitconvert import *


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
