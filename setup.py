#!/usr/bin/env python

from setuptools import setup

with open('README.MD') as f:
	long_description = f.read()

reqs = open('requirements.txt').readlines()

setup(
	name='sdsmc', #NOTE: dsmc was already taken as a pypi name
	version='1.0.0',
	description='Sandia Distribution System Model Calibration Algorithms',
	long_description_content_type='text/markdown',
	long_description=long_description,
	author='Logan Blakely',
	author_email='lblakel@sandia.gov',
	url='https://github.com/sandialabs/distribution-system-model-calibration',
	packages = ['sdsmc'],
	include_package_data=True,
	setup_requires=reqs,
	install_requires=reqs
)