from setuptools import setup, find_packages

setup(
    name="nmp-modeling",
    version="0.0.1",
    packages=find_packages(include=["nmp_modeling", "nmp_modeling.*"]),
    extras_require={"optimization": ["cmaes>=0.13,<0.14"]},
)
