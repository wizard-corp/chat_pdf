#!/bin/sh

set -e
echo "STARTING..."

(
    python src/app.py
    # Exit with the appropriate code
    exit_code=$?
    echo "Server exited with code: $exit_code"
    exit $exit_code
)

exec "$@"
