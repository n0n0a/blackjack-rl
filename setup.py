#!/usr/bin/envs python
from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "gym>=0.18.0",
    "numpy>=1.21.0",
    "pytest>=6.2.4",
    "scipy>=1.6.3",
    "seaborn>=0.11.1"
]

setup(
    name='blackjack_rl',
    version="0.0.1",
    description='Reinforcement Learning for Blackjack Game',
    author='n0n0a',
    url="https://github.com/n0n0a/blackjack-rl",
    license='MIT',
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(exclude=["test"]),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License"
    ]
)
