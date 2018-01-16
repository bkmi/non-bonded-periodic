from setuptools import setup


setup(
    name="non-bonded-periodic",
    version="0.1.0",
    author="Alexy, Ben, Chris, Ludovica, Tracy",
    author_email="",
    description="A module for doing mcmc on a box of non-bonded particles with periodic boundary conditions.",
    license="MIT",
    keywords="mcmc markov chain monte carlo molecule",
    url="https://github.com/machism0/non-bonded-periodic",
    packages=['nbp', 'nbp.tests'],
    setup_requires=['pytest-runner'],
    install_requires=['matplotlib', 'numpy', 'scipy', 'seaborn', 'pathlib'],
    tests_require=['pytest']
)
