# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup Xee."""
import setuptools

# # TODO(alxr): Add docs support.
# docs_requires = [
#     'myst-nb',
#     'myst-parser',
#     'sphinx',
#     'sphinx_rtd_theme',
#     'scipy',
# ]
tests_requires = [
    'absl-py',
    'pytest',
    'pyink',
]

examples_require = [
    'xarray-beam',
    'absl-py',
]

setuptools.setup(
    name='xee',
    version='0.0.0',
    license='Apache 2.0',
    author='Google LLC',
    author_email='noreply@google.com',
    install_requires=['xarray', 'earthengine-api', 'pyproj', 'affine'],
    extras_require={
        'tests': tests_requires,
        'examples': examples_require,
    },
    url='https://github.com/google/xee',
    packages=setuptools.find_packages(exclude=['examples']),
    python_requires='>=3',
    entry_points={
        'xarray.backends': ['ee=xee:EarthEngineBackendEntrypoint'],
    }
)
