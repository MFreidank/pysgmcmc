from os.path import realpath, dirname, join as path_join
from setuptools import setup, find_packages

NAME = "pysgmcmc"
DESCRIPTION = "PySGMCMC"
LONG_DESCRIPTION = "PYSGMCMC is a Python framework for Bayesian Deep Learning which focuses on Stochastic Gradient Markov Chain Monte Carlo methods."
MAINTAINER = "Moritz Freidank"
MAINTAINER_EMAIL = "freidankm@googlemail.com"
URL = "https://github.com/MFreidank/pysgmcmc"
# license = ??
VERSION = "0.0.1"

PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = path_join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE, "r") as f:
    requirements = f.read().splitlines()

INSTALL_REQUIREMENTS = [
    requirement for requirement in requirements
    if not requirement.startswith("http")
]

DEPENDENCY_LINKS = [
    requirement for requirement in requirements if requirement.startswith("http")

]

SETUP_REQUIREMENTS = ["pytest-runner"]
TEST_REQUIREMENTS = ["pytest", "pytest-cov"]


if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        url=URL,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        package_data={"docs": ["*"]},
        include_package_data=True,
        install_requires=INSTALL_REQUIREMENTS,
        dependency_links=DEPENDENCY_LINKS,
        setup_requires=SETUP_REQUIREMENTS,
        tests_require=TEST_REQUIREMENTS,
    )
