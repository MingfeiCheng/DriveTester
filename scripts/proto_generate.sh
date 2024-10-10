#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

APOLLO_PROTO_PATH="$PARENT_DIR/Apollo/"
APOLLO_PROTO_OUT_PATH="$PARENT_DIR"

# Find all .proto files, excluding .cache directories
find "$APOLLO_PROTO_PATH" -name '*.proto' | while read -r proto_file; do
    # Check if the file is inside a .cache directory
    if [[ "$proto_file" == *".cache"* ]]; then
        continue
    fi

    # Compile the .proto file
    protoc -I="$APOLLO_PROTO_PATH" --python_out="$APOLLO_PROTO_OUT_PATH" "$proto_file"
done