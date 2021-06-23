#!/usr/bin/env python
from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "numpy>=1.21.0",
]

setup(
    name='blackjack-rl',
    version="0.0.1",
    description='reinforcement learning for blackjack game',
    author='n0n0a',
    url="https://github.com/n0n0a/blackjack-rl",
    license='MIT',
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License"
    ]
)
