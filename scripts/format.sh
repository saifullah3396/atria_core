#!/usr/bin/env bash

set -e
set -x

ruff check src --fix      # linter
ruff format src --check # formatter
