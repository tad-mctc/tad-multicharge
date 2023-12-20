# This file is part of tad-multicharge.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2023 Marvin Friede
#
# tad-multicharge is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad-multicharge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-multicharge. If not, see <https://www.gnu.org/licenses/>.
"""
Config file for docs.
"""
import os.path as op
import sys

sys.path.insert(0, op.join(op.dirname(__file__), "..", "src"))

import tad_multicharge

project = "Torch Autodiff Multicharge"
author = "Marvin Friede"
copyright = f"2023 {author}"

extensions = [
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

html_theme = "sphinx_book_theme"
html_title = project
html_logo = "_static/tad_multicharge.svg"
html_favicon = "_static/tad_multicharge-favicon.svg"

html_theme_options = {
    "repository_url": "https://github.com/tad-mctc/tad-multicharge",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_download_button": False,
    "path_to_docs": "doc",
    "show_navbar_depth": 3,
    "logo_only": False,
}

html_sidebars = {}  # type: ignore[var-annotated]

html_css_files = [
    "css/custom.css",
]
html_static_path = ["_static"]
templates_path = ["_templates"]

autodoc_typehints = "none"
autosummary_generate = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

master_doc = "index"
