#!/usr/bin/env bash

set -e
set -x

coverage run --source=atria_core -m pytest $@
coverage report --show-missing
