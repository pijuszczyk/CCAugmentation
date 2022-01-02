"""
CCAugmentation

Data preprocessing & augmentation framework that is designed for working with crowd counting datasets.
Supports multitude of simple as well as advanced transformations, outputs and loaders, all of them to be combined using
pipelines.
"""

__title__ = "CCAugmentation"
__version__ = "0.1.0"
__author__ = "Piotr Juszczyk"
__license__ = "MIT"

from . import integrations
from . import loaders
from . import operations
from . import outputs
from .pipelines import Pipeline
from . import transformations

__all__ = ["integrations", "loaders", "operations", "outputs", "pipelines", "transformations"]
