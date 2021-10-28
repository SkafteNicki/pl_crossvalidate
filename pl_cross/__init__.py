#!/usr/bin/env python
"""Root package info."""

import logging as python_logging
import os
import time

# fmt: off
__name__ = "pl_cross"
_this_year = time.strftime("%Y")
__version__ = "0.1.0"
__author__ = "Nicki Skafte Detlefsen"
__author_email__ = "skaftenicki@gmail.com"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2021-{_this_year}, {__author__}."
__homepage__ = "https://github.com/SkafteNicki/pl_cross"
__docs__ = "Cross validation in pytorch lightning made easy"
# fmt: on

_logger = python_logging.getLogger("pl_cross")
_logger.addHandler(python_logging.StreamHandler())
_logger.setLevel(python_logging.INFO)

PACKAGE_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

try:
    # This variable is injected in the __builtins__ by the build process
    _ = None if __PL_CROSS__ else None
except NameError:
    __PL_CROSS__: bool = False

if __PL_CROSS__:  # pragma: no cover
    import sys

    sys.stdout.write(f"Partial import of `{__name__}` during the build process.\n")
    # We are not importing the rest of the package during the build process, as it may not be compiled yet
else:
    # import modules
    from .trainer import Trainer
    from .datamodule import BaseKFoldDataModule, KFoldDataModule