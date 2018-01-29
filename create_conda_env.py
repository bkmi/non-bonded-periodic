import subprocess
import os


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
