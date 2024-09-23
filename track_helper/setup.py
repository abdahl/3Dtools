'''
Install with  pip install -e /path/to/insegtpy/folder/containing/this/file/
-e stands for editable,  so it will then create a link from the site-packages 
directory to the directory in which the code lives, meaning the latest version 
will be used without need to reinstall.

Following info from: https://stackoverflow.com/a/50468400 and
https://python-packaging.readthedocs.io/en/latest/index.html

'''
from setuptools import setup

setup(name='track_helper',
    version='0.1',
    description='Tracking of ud fibers.',
    url='https://github.com/abdahl/3Dtools',
    author='Anders Dahl',
    author_email='abda@dtu.dk',
    license='GNU GPSv3',
    install_requires=[
          'tifffile',
          'numpy',
          'matplotlib'
      ],
    zip_safe=False)