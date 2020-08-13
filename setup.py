from setuptools import setup, find_packages

with open('README.md') as f:
  readme = f.read()

setup(
    name='tracklib',
    version='0.0.0',
    description='A library for analyzing two-locus tracking data',
    long_description=readme,
    author='Simon Grosse-Holz',
    packages=find_packages(exclude=('tests', 'docs')),
    )
