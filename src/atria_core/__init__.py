"""
Core Module

This module contains the core utilities and components of the Atria framework. It includes logging, registry management, and other foundational utilities.

Submodules:
    - logger: Logging utilities for the Atria framework.
    - registry: Registry management for organizing and accessing components.
    - utilities: Common utility functions and classes.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from .utilities.yaml_resolvers import _dir_name_from_overrides  # noqa: F401

__version__ = "0.1.0"
