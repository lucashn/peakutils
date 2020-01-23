from setuptools import setup

with open('README.rst') as readme:
    long_description = readme.read()

setup(
    name='PeakUtils',
    version='1.3.3',
    description='Peak detection utilities for 1D data',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author='Lucas Hermann Negri',
    author_email='lucashnegri@gmail.com',
    url='https://bitbucket.org/lucashnegri/peakutils',
    packages=['peakutils'],
    install_requires=['numpy', 'scipy'],
    tests_require=['pandas'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    license='MIT',
    test_suite='tests',
    keywords='peak detection search gaussian centroid baseline maximum',
)
