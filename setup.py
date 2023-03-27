'''
Date: 2022-07-20 15:22:19
LastEditors: mushan wwd137669793@gmail.com
LastEditTime: 2023-03-25 16:49:42
FilePath: /mushan-pipy/setup.py
'''
import os
import sys
from setuptools import setup
from setuptools import find_packages

NAME = "mushan"
AUTHOR = "Mushan"
EMAIL = "wwd137669793@gmail.com"
URL = "None"
LICENSE = "None"
DESCRIPTION = "Personal toolkit."
VERSION = "0.0.11"

if sys.version_info < (3, 6, 0):
    raise RuntimeError(f"{NAME} requires Python >=3.6.0, but yours is {sys.version}!")

__version__ = VERSION

try:
    with open("README.md", encoding="utf8") as f_r:
        _long_description = f_r.read()
except FileNotFoundError:
    _long_description = ""

if __name__ == "__main__":
    setup(
        name=NAME,
        version=__version__,
        author=AUTHOR,
        url="https://github.com/mushanshanshan/mushan-pip",
        author_email=EMAIL,
        Homepage=URL,
        license=LICENSE,
        description=DESCRIPTION,
        packages=find_packages(),
        include_package_data=True,
        setup_requires=["setuptools>=18.0", "wheel"],
        install_requires=open("./requirements.txt", "r").read().splitlines(),
        long_description=_long_description,
        long_description_content_type="text/markdown",
        entry_points={
            "console_scripts": [
                "mushan=mushan.shell:run"
            ]
        },
        package_data={
            "mushan": ["src/*.txt"]
        },
        zip_safe=True,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.6"
    )
