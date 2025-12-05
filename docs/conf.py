# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))

import torchgeo

# -- Project information -----------------------------------------------------

project = 'torchgeo'
copyright = 'TorchGeo Contributors'
author = torchgeo.__author__
version = '.'.join(torchgeo.__version__.split('.')[:2])
release = torchgeo.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']

# Sphinx 5.3+ required to allow section titles inside autodoc class docstrings
# https://github.com/sphinx-doc/sphinx/pull/10887
needs_sphinx = '5.3'

nitpicky = True
nitpick_ignore = [
    # Undocumented classes
    ('py:class', 'kornia.augmentation._2d.intensity.base.IntensityAugmentationBase2D'),
    ('py:class', 'kornia.augmentation._3d.geometric.base.GeometricAugmentationBase3D'),
    ('py:class', 'kornia.augmentation.base._AugmentationBase'),
    ('py:class', 'lightning.pytorch.utilities.types.LRSchedulerConfig'),
    ('py:class', 'lightning.pytorch.utilities.types.OptimizerConfig'),
    ('py:class', 'lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig'),
    ('py:class', 'segmentation_models_pytorch.base.model.SegmentationModel'),
    ('py:class', 'timm.models.resnet.ResNet'),
    ('py:class', 'timm.models.vision_transformer.VisionTransformer'),
    ('py:class', 'torch.optim.lr_scheduler.LRScheduler'),
    ('py:class', 'torchvision.models._api.WeightsEnum'),
    ('py:class', 'torchvision.models.resnet.ResNet'),
    ('py:class', 'torchvision.models.swin_transformer.SwinTransformer'),
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'furo'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation: https://pradyunsg.me/furo/.
html_theme_options = {
    # Sidebar
    'sidebar_hide_name': False,
    'navigation_with_keys': True,
    # Light/dark mode colors
    'light_css_variables': {
        'color-brand-primary': '#0071bc',
        'color-brand-content': '#0071bc',
    },
    'dark_css_variables': {
        'color-brand-primary': '#5b9bd5',
        'color-brand-content': '#5b9bd5',
    },
    # GitHub repository
    'source_repository': 'https://github.com/torchgeo/torchgeo',
    'source_branch': 'main',
    'source_directory': 'docs/',
    # Footer
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/torchgeo/torchgeo',
            'html': """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            'class': '',
        },
        {
            'name': 'Slack',
            'url': 'https://torchgeo.slack.com/join/shared_invite/zt-36dtc53qq-D9G35xRSQC702aX7x_PK~A#/shared-invite/email',
            'html': """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16" style="margin-left: 0.5rem;">
                    <path d="M3.362 10.11c0 .926-.756 1.681-1.681 1.681S0 11.036 0 10.111C0 9.186.756 8.43 1.68 8.43h1.682v1.68zm.846 0c0-.924.756-1.68 1.681-1.68s1.681.756 1.681 1.68v4.21c0 .924-.756 1.68-1.68 1.68a1.685 1.685 0 0 1-1.682-1.68v-4.21zM5.89 3.362c-.926 0-1.682-.756-1.682-1.681S4.964 0 5.89 0s1.68.756 1.68 1.68v1.682H5.89zm0 .846c.924 0 1.68.756 1.68 1.681S6.814 7.57 5.89 7.57H1.68C.757 7.57 0 6.814 0 5.89c0-.926.756-1.682 1.68-1.682h4.21zm6.749 1.682c0-.926.755-1.682 1.68-1.682s1.68.756 1.68 1.681-.756 1.681-1.68 1.681h-1.681V5.89zm-.848 0c0 .924-.755 1.68-1.68 1.68A1.685 1.685 0 0 1 8.43 5.89V1.68C8.43.757 9.186 0 10.11 0c.926 0 1.681.756 1.681 1.68v4.21zm-1.681 6.748c.926 0 1.682.756 1.682 1.681S11.036 16 10.11 16s-1.681-.756-1.681-1.68v-1.682h1.68zm0-.847c-.924 0-1.68-.755-1.68-1.68 0-.925.756-1.681 1.68-1.681h4.21c.924 0 1.68.756 1.68 1.68 0 .926-.756 1.681-1.68 1.681h-4.21z"/>
                </svg>
            """,
            'class': '',
        },
    ],
}

html_logo = os.path.join('..', 'logo', 'logo-color.svg')
html_favicon = os.path.join('..', 'logo', 'favicon.ico')

html_static_path = ['_static']
html_css_files = ['badge-height.css', 'notebook-prompt.css', 'table-scroll.css']

# -- Extension configuration -------------------------------------------------

# sphinx.ext.autodoc
autodoc_default_options = {
    'members': True,
    'special-members': True,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# sphinx.ext.intersphinx
intersphinx_mapping = {
    'einops': ('https://einops.rocks/', None),
    'kornia': ('https://kornia.readthedocs.io/en/stable/', None),
    'lightning': ('https://lightning.ai/docs/pytorch/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'pyproj': ('https://pyproj4.github.io/pyproj/stable/', None),
    'python': ('https://docs.python.org/3', None),
    'rasterio': ('https://rasterio.readthedocs.io/en/stable/', None),
    'segmentation_models_pytorch': ('https://smp.readthedocs.io/en/stable/', None),
    'shapely': ('https://shapely.readthedocs.io/en/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'timm': ('https://huggingface.co/docs/timm/main/en/', None),
    'torch': ('https://docs.pytorch.org/docs/stable/', None),
    'torchmetrics': ('https://lightning.ai/docs/torchmetrics/stable/', None),
    'torchvision': ('https://docs.pytorch.org/vision/stable/', None),
}

# nbsphinx
nbsphinx_execute = 'never'
with open(os.path.join('tutorials', 'prolog.rst.jinja')) as f:
    nbsphinx_prolog = f.read()
