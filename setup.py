#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import os

from setuptools import setup, find_packages

try:
    with open('README.rst') as f:
        readme = f.read()
except IOError:
    readme = ''


def _requires_from_file(filename):
    return open(filename).read().splitlines()


# version
here = os.path.dirname(os.path.abspath(__file__))
version = "0.0.4"

setup(
    name="novograd",
    version=version,
    url='https://github.com/convergence-lab/novograd',
    author='Masashi Kimura',
    author_email='kimura@convergence-lab.com',
    maintainer='Masashi Kimura',
    maintainer_email='kimura@convergence-lab.com',
    description='PyTorch implementation of NovoGrad',
    long_description=readme,
    packages=find_packages(),
    install_requires=[
        "torch >= 1.2.0"
    ],
    license="Apache License 2.0",
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
    ],
)
