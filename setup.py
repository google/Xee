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
    'apache_beam[gcp]',
    'xarray-beam',
    'absl-py',
    'gcsfs',
]

setuptools.setup(
    name='xee',
    version='0.0.0',
    license='Apache 2.0',
    author='Google LLC',
    author_email='noreply@google.com',
    description='A Google Earth Engine extension for Xarray.',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    install_requires=['xarray', 'earthengine-api>=0.1.374', 'pyproj', 'affine'],
    extras_require={
        'tests': tests_requires,
        'examples': examples_require,
    },
    url='https://github.com/google/xee',
    packages=setuptools.find_packages(exclude=['examples']),
    python_requires='>=3.9',
    entry_points={
        'xarray.backends': ['ee=xee:EarthEngineBackendEntrypoint'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Atmospheric Science',

    ],
    project_urls={
        'Issue Tracking': 'https://github.com/google/Xee/issues',
    },
)
