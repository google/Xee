"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup ----------------------------------------------------------------
import subprocess
import sys
import xee

# -- Project information -------------------------------------------------------
project = "Xee"
copyright = "2023, Google LCC"
author = "The Xee authors"
# wait for https://github.com/google/Xee/pull/162
# release = xee.__version__

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]

# -- Options for intersphinx output --------------------------------------------
intersphinx_mapping = {
    "xarray": ("https://xarray.pydata.org/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/xee-logo.png"
html_favicon = "_static/xee-logo.png"

# -- Options for autosummary/autodoc output ------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
