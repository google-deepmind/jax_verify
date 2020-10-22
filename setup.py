# coding=utf-8
# Copyright 2020 The jax_verify Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for pip package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from setuptools import find_namespace_packages
from setuptools import setup


def _parse_requirements(requirements_txt_path):
  with open(requirements_txt_path) as fp:
    return fp.read().splitlines()


def test_suite():
  test_loader = unittest.TestLoader()
  all_tests = test_loader.discover('jax_verify/tests',
                                   pattern='*_test.py')
  return all_tests

setup(
    name='jax_verify',
    version='1.0',
    description='A library for neural network verification.',
    url='https://github.com/deepmind/jax_verify',
    author='DeepMind',
    author_email='jax_verify@google.com',
    # Contained modules and scripts.
    packages=find_namespace_packages(exclude=['*_test.py']),
    install_requires=_parse_requirements('requirements.txt'),
    requires_python='>=3.6',
    platforms=['any'],
    license='Apache 2.0',
    test_suite='setup.test_suite',
    include_package_data=True,
    zip_safe=False,
)
