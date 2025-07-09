#!/usr/bin/env bash

set -e
set -x

uv run coverage run --source=atria_core -m pytest $@ 
uv run coverage report --show-missing
