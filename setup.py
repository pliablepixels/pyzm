#!/usr/bin/python3

import io
import os
import re
import codecs

from setuptools import setup, find_packages

#Package meta-data.
NAME = 'pyzm'
DESCRIPTION = 'ZoneMinder API, Logger and other base utilities for python programmers'
URL = 'https://github.com/pliablepixels/pyzm'
AUTHOR_EMAIL = 'info@zoneminder.com'
AUTHOR = 'Pliable Pixels'
LICENSE = 'GPL'
INSTALL_REQUIRES=[
    'requests>=2.18.4',
    'pydantic>=2.0.0',
    'dateparser>=1.1.0',
    'mysql-connector-python>=8.0.16',
    'python-dotenv',
    ]

_ML_REQUIRES=[
    'numpy>=1.13.3',
    'Pillow',
    'onnx>=1.12.0',
    'Shapely>=1.7.0',
    'portalocker>=2.3.0',
    ]


here = os.path.abspath(os.path.dirname(__file__))
# read the contents of your README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    f.close()

def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        data = fp.read()
        fp.close()
        return data

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(name = NAME,
      python_requires='>=3.0.0',
      version = find_version('pyzm','__init__.py'),
      description = DESCRIPTION,
      long_description = long_description,
      long_description_content_type='text/markdown',
      author = AUTHOR,
      author_email = AUTHOR_EMAIL,
      url = URL,
      project_urls={
          'Documentation': 'https://pyzmv2.readthedocs.io/en/latest/',
          'Source': 'https://github.com/pliablepixels/pyzm',
          'Bug Tracker': 'https://github.com/pliablepixels/pyzm/issues',
      },
      license = LICENSE,
      install_requires=INSTALL_REQUIRES,
      extras_require={
          'ml': _ML_REQUIRES,
          'serve': _ML_REQUIRES + [
              'fastapi>=0.100',
              'uvicorn>=0.20',
              'python-multipart>=0.0.5',
              'PyJWT>=2.0',
              'PyYAML>=5.0',
          ],
          'train': _ML_REQUIRES + [
              'ultralytics>=8.3',
              'streamlit>=1.41',
              'streamlit-drawable-canvas>=0.9',
              'st-clickable-images>=0.0.3',
              'PyYAML>=5.0',
          ],
      },
      packages=find_packages(exclude=["tests", "tests.*"]),
      )

