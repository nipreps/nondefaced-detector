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
import os
import sys

import sphinx_rtd_theme

from datetime import date

sys.path.insert(0, os.path.abspath("../"))

import nondefaced_detector


# -- Project information -----------------------------------------------------

project = "nondefaced-detector"
copyright = "2021-{}, Developers of Nondefaced-Detector".format(date.today().year)
author = "Developers of Nondefaced-Detector"

currentdir = os.path.abspath(os.path.dirname(__file__))
currentdir = os.path.abspath(os.path.dirname(__file__))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "numpydoc",
    "sphinx_rtd_theme",
]

intersphinx_mapping = {
    "https://docs.python.org/3.5": None,
    "https://docs.scipy.org/doc/numpy": None,
    "https://docs.scipy.org/doc/scipy/reference": None,
    "https://matplotlib.org/": None,
    "https://scikit-learn.org/0.17": None,
    "https://nipy.org/nibabel/": None,
    "https://pandas.pydata.org/pandas-docs/stable/": None,
    "https://neurosynth.readthedocs.io/en/latest/": None,
}

intersphinx_timeout = 5

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The short X.Y version.
version = nondefaced_detector.__version__
# The full version, including alpha/beta/rc tags.
release = version

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ("index", "nondefaced_detector", "nondefaced-detector Documentation", [author], 1)
]

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "nondefaced_detector",
        "nondefaced-detector Documentation",
        author,
        "nondefaced_detector",
        "One line description of project.",
        "Miscellaneous",
    )
]
