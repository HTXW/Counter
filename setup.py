"""
Created on 14. 9. 2012.

@author: kermit
"""

import os

from setuptools import setup, find_packages
from Cython.Build import cythonize


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="philharmonic",
    version="0.1",
    packages=find_packages(),
    install_requires=['pandas>=2.2.2',
                      'numpy>=1.24.3',
                      'pysnmp>=4.4.12',
                      'SOAPpy>=0.12.22',
                      'twisted>=22.10.0'],
    #  sudo apt-get intsall python-twisted-web
    #  ext_modules = cythonize('**/*.pyx'),
    zip_safe=True,

    entry_points={
        'console_scripts': ['ph=simulate:cli', 'philharmonic=simulate:cli']
    }
)
