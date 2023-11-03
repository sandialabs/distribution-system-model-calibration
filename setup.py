#!/usr/bin/env python

from setuptools import setup

with open('README.MD') as f:
	long_description = f.read()
	
reqs = open('requirements.txt').readlines()
	
setup(
	name='snl_dsmc',
	version='1.0.0',
	description='SNL Distribution System Model Calibration Algorithms',
	long_description_content_type='text/markdown',
	long_description=long_description,
	author='Logan Blakely',
	author_email='lblakel@sandia.gov',
	url='https://github.com/sandialabs/distribution-system-model-calibration',
	packages = ['snl_dsmc'],
	include_package_data=True,
	setup_requires=reqs,
	install_requires=reqs
)