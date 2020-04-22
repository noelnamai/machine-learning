#!/usr/bin/env python3

from setuptools import setup

setup(
    name = "sebaba", 
    version = "0.0.1",
    description = "A bare metal implementation of Machine Learning algorithms",
    author = "Noel Namai",
    author_email = "noelnamai@yahoo.com",
    url = "http://github.com/noelnamai/sebaba/",
    license = "MIT",
    package = ["sebaba", "sebaba.machinelearning"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.6",
)