#! usr/bin/python
from distutils.core import setup
import os


setup(
    name='supergametools',
    version='0.1',
    author='Benjamin Tengelsen',
    author_email='btengels@cmu.edu',
    packages=['supergametools'],
    url='https://bitbucket.org/ben_tengelsen/supergametools',
    description='Python library for finding equilibria of repeated games',
    long_description=open('README.txt').read()
)
