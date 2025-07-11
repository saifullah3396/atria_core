#!/usr/bin/env bash

set -e
set -x

mypy src --follow-imports=skip     # type check
ruff check src        # linter
ruff format src --check # formatter
