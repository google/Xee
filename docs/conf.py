# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# Print Python environment info for easier debugging on ReadTheDocs

import subprocess
import sys
import xee  # verify this works

print('python exec:', sys.executable)
print('sys.path:', sys.path)
print('pip environment:')
subprocess.run([sys.executable, '-m', 'pip', 'list'])  # pylint: disable=subprocess-run-check

print(f'xee: {xee.__file__}')

# -- Project information -----------------------------------------------------

project = 'Xee'
copyright = '2023, Google LCC'  # pylint: disable=redefined-builtin
author = 'The Xee authors'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'myst_nb',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_templates', 'Thumbs.db', '.DS_Store', 'README.md', 'user-guide.md']

intersphinx_mapping = {
    'xarray': ('https://xarray.pydata.org/en/latest/', None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Keep the left-hand navigation consistent on every page. In particular,
# - titles_only=True prevents page section headings (e.g., "Goals", "Approach")
#   from appearing in the sidebar when viewing that page.
# - collapse_navigation=False expands the full toctree instead of focusing only
#   on the current page branch (which makes the list look different per page).
# - sticky_navigation=False avoids auto-scrolling the sidebar to keep the
#   current entry near the top, which can give the impression of different
#   ordering between pages.
# - navigation_depth controls how deep the tree expands. With titles_only=True,
#   this is the depth of documents, not their internal sections.
html_theme_options = {
    'titles_only': True,
    'collapse_navigation': False,
    'sticky_navigation': False,
    # Do not include hidden local toctrees (e.g., autosummary children) in the
    # sidebar, and limit the sidebar to top-level only.
    'includehidden': False,
    'navigation_depth': 1,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Extension config

autosummary_generate = True

# https://myst-nb.readthedocs.io/en/latest/computation/execute.html#notebook-execution-modes
nb_execution_mode = 'off'
# https://myst-nb.readthedocs.io/en/latest/render/format_code_cells.html#remove-stdout-or-stderr
nb_output_stderr = 'remove-warn'

# https://stackoverflow.com/a/66295922/809705
autodoc_typehints = 'description'

# Note: We exclude README.md from Sphinx to avoid duplicate toctree references,
# since the landing page content is provided by index.md.
