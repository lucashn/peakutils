from setuptools import setup, Command


class my_clean(Command):

    description = "Removes the generated files from the directory"
    user_options = []

    def initialize_options(self):
        pass

    def run(self):
        import os
        import shutil

        try:
            os.remove('MANIFEST')
        except:
            pass

        dirs = ['peakutils/__pycache__', 'PeakUtils.egg-info', 'build', 'dist']

        for dir in dirs:
            shutil.rmtree(dir, True)

    def finalize_options(self):
        pass

with open('README.rst') as readme:
    long_description = readme.read()

setup(
    name='PeakUtils',
    version='1.1.0',
    description='Peak detection utilities for 1D data',
    author='Lucas Hermann Negri',
    author_email='lucashnegri@gmail.com',
    url='https://bitbucket.org/lucashnegri/peakutils',
    packages=['peakutils'],
    install_requires=['numpy', 'scipy'],
    cmdclass={
        'clean': my_clean,
    },
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
