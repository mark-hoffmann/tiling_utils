import os
import os.path
import sys
from setuptools import setup, find_packages

PACKAGE_NAME = 'tiling_utils'
MINIMUM_PYTHON_VERSION = 3, 6


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    raise KeyError("'{0}' not found in '{1}'".format(key, module_path))


def read_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf8') as file:
        return file.read()


check_python_version()
version = read_package_variable('__version__')

setup(
    name=PACKAGE_NAME,
    version=version,
    description='Utilities for working with large images for deep learning applications',
    author='Mark Hoffmann',
    packages=find_packages(exclude=['contrib', 'docs', 'site', 'tests*']),
    install_requires=[
        'torch',
        'torchvision',
    ],
    tests_require=[
        'pytest>=4.3.1',
    ],
    url='https://github.com/mark-hoffmann/tiling_utils',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
)
