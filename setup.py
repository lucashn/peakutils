try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.rst') as readme:
    long_description = readme.read()

import numpy

setup(
    name='PeakUtils',
    version='0.1.0',
    description='Peak detection utilities',
    author='Lucas Hermann Negri',
    author_email='lucashnegri@gmail.com',
    url='https://bitbucket.org/lucashnegri/peakutils',
    packages=['peakutils'],
    install_requires=['numpy', 'scipy', 'matplotlib'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    license="MIT"
 )

# ahh, so clean
import sys
import os
import glob
import shutil

if "clean" in sys.argv:
    print("removing junk")

    os.remove("MANIFEST")
    dirs = ["peakutils/__pycache__", "peakutils/__pycache__", "PeakUtils.egg-info",
            "build", "dist"]

    for dir in dirs:
        shutil.rmtree(dir, True)
