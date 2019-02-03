#! /usr/bin/env bash

# Exit on error
set -e
# Echo each command
set -x

echo "Compile extensions"
make
pytest
