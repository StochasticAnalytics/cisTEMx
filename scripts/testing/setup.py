from setuptools import setup, find_packages

setup(
    name="cistem_test_utils",
    version="0.1.0",
    packages=find_packages(where="programs"),
    package_dir={"": "programs"},
    install_requires=[
        "mrcfile",
        "toml",
        "numpy"
    ],
    description="Testing utilities for cisTEM",
)