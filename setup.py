import sys
import subprocess
import os

from setuptools import setup
from setuptools.command.test import test as testcommand

# from pathlib import Path
#
# env = Path('environment.yml')

if os.name != 'nt':
    # UNIX/MAC
    try:
        with open(os.devnull, 'wb') as quiet:
            subprocess.run('conda env create -f environment.yml'.split(),
                           check=True,
                           stderr=quiet)
    except subprocess.CalledProcessError:
        subprocess.run('conda env update -f environment.yml'.split())
else:
    # WINDOWS
    try:
        with open(os.devnull, 'wb') as quiet:
            subprocess.run('conda env create -f win-environment.yml'.split(),
                           check=True,
                           stderr=quiet)
    except subprocess.CalledProcessError:
        subprocess.run('conda env update -f win-environment.yml'.split())


class PyTest(testcommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        testcommand.initialize_options(self)
        self.pytest_args = ['nbp']

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    cmdclass={'test': PyTest},
    name="non-bonded-periodic",
    version="0.1.0",
    author="",
    author_email="",
    description="A module for doing mcmc on a box of non-bonded particles with periodic boundary conditions.",
    license="MIT",
    keywords="mcmc markov chain monte carlo molecule",
    url="https://github.com/machism0/non-bonded-periodic",
    packages=['nbp', 'nbp.test'],
    setup_requires=['pytest-runner'],
    install_requires=['matplotlib', 'numpy', 'scipy', 'seaborn', 'pathlib'],
    tests_require=['pytest']
)
