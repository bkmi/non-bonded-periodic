from setuptools import setup
import subprocess
import os

from pathlib import Path

env = Path('environment.yml')

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

setup(
    name="non-bonded-periodic",
    version="0.0.0",
    author="",
    author_email="",
    description="A module for doing mcmc on a box of non-bonded particles with periodic boundary conditions.",
    license="MIT",
    keywords="mcmc markov chain monte carlo molecule",
    url="https://github.com/machism0/non-bonded-periodic",
    packages=['an_example_pypi_project', 'tests'],
    long_description="",
    classifiers=[],
)
