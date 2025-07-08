#!/usr/bin/env bash

set -e
set -x

ruff check $@        # linter
ruff format $@ --check # formatter
