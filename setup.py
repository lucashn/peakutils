from setuptools import setup

with open('README.rst') as readme:
    long_description = readme.read()

setup(
    name='PeakUtils',
    version='0.2.0',
    description='Peak detection utilities for 1D data',
    author='Lucas Hermann Negri',
    author_email='lucashnegri@gmail.com',
    url='https://bitbucket.org/lucashnegri/peakutils',
    packages=['peakutils'],
    install_requires=['numpy', 'scipy'],
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

    try:
        os.remove("MANIFEST")
    except:
        pass
    
    dirs = ["peakutils/__pycache__", "peakutils/__pycache__", "PeakUtils.egg-info",
            "build", "dist"]

    for dir in dirs:
        try:
            shutil.rmtree(dir, True)
        except:
            pass
