"""
File Utilities Module

This module provides utility functions for handling file paths and resolving
them. It includes functionality for constructing full paths and optionally
validating their existence.

Functions:
    - _resolve_path: Constructs a full path by concatenating path segments and optionally validates its existence.

Dependencies:
    - pathlib.Path: For handling and resolving file paths.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _resolve_path(*args: str, validate: bool = True) -> Path:
    """
    Constructs a full path by concatenating all arguments and optionally validates its existence.

    Args:
        *args (str): Path segments to concatenate.
        validate (bool): If True, checks whether the constructed path exists.

    Returns:
        Path: The full constructed path.

    Raises:
        FileNotFoundError: If `validate=True` and the path does not exist.
    """

    from pathlib import Path

    full_path = Path(*args).resolve()

    if validate and not full_path.exists():
        raise FileNotFoundError(f"Path does not exist: {full_path}")

    return full_path


def _load_bytes_from_uri(uri: str) -> bytes:
    """
    Load raw bytes from the given URI.
    Supports:
        - Local files
        - tar:// URIs with offset & length query (local)
        - HTTP/HTTPS/FTP URLs (full file or partial with offset/length)
    """

    from pathlib import Path
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    import requests

    parsed = urlparse(uri)
    query = parse_qs(parsed.query)
    path = parsed.path

    if parsed.scheme in ["http", "https", "ftp"]:
        if path.endswith(".tar"):
            try:
                offset = int(query.get("offset", [None])[0])
                length = int(query.get("length", [None])[0])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Missing or invalid 'offset' and 'length' in tar URI: {uri}"
                )
            # Remove offset and length from query dict
            query.pop("offset", None)
            query.pop("length", None)

            # Rebuild the query string without offset and length
            new_query = urlencode(query, doseq=True)

            # Rebuild the URL without offset and length params
            new_url = urlunparse(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    parsed.params,
                    new_query,
                    parsed.fragment,
                )
            )

            response = requests.get(
                new_url,
                headers={"Range": f"bytes={offset}-{offset + length - 1}"},
                stream=True,
            )
            response.raise_for_status()
            return response.content
        else:
            response = requests.get(uri)
            response.raise_for_status()
            return response.content

    elif parsed.scheme in ["", "file", "tar"]:
        # local file or tar file with offset/length
        if path.endswith(".tar"):
            try:
                offset = int(query.get("offset", [None])[0])
                length = int(query.get("length", [None])[0])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Missing or invalid 'offset' and 'length' in tar URI: {uri}"
                )

            local_path = Path(path)
            if not local_path.exists():
                raise FileNotFoundError(f"TAR archive not found: {local_path}")

            with open(local_path, "rb") as f:
                f.seek(offset)
                bytes = f.read(length)
                return bytes
        else:
            local_path = Path(path if parsed.scheme != "file" else parsed.path)
            if not local_path.exists():
                raise FileNotFoundError(f"File not found: {local_path}")
            if not local_path.is_file():
                raise ValueError(f"Provided path is not a file: {local_path}")

            return local_path.read_bytes()
    else:
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")
