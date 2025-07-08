#!/usr/bin/env bash

set -e
set -x

mypy $@            # type check
ruff check $@        # linter
ruff format $@ --check # formatter
