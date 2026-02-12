"""
DataSentry
==========

A lightweight, production-ready, data-centric machine learning inspection library.

Main entry point:
    analyze()

Example:
--------
>>> from datasentry import analyze
>>> report = analyze(X, y)
>>> report.show()
"""

from .analyzer import analyze
from .config import DataSentryConfig

__all__ = ["analyze", "DataSentryConfig"]

__version__ = "0.1.0"
