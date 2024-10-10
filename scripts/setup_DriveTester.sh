#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

DEST_PATH="$PARENT_DIR/Apollo"
APOLLO_PROTO_PATH="$PARENT_DIR/Apollo/"
APOLLO_PROTO_OUT_PATH="$PARENT_DIR"
ENV_EXPORT_PATH="$PARENT_DIR/environment.yml"
#
# Step 1: Clone the Apollo repository (version 7.0.0) into the specified directory
echo "Cloning Apollo repository (v7.0.0) into $DEST_PATH..."
git clone -b v7.0.0 https://github.com/ApolloAuto/apollo.git "$DEST_PATH"

cp $PARENT_DIR/scripts/WORKSPACE $DEST_PATH/WORKSPACE

# Step 2: Change directory to the Apollo folder
cd "$DEST_PATH" || exit

# Step 3: Start the Apollo Docker container
echo "Starting Apollo Docker container..."
bash docker/scripts/dev_start.sh

# Step 4: Build Apollo inside the container
echo "Building Apollo inside the container..."
docker exec -it --user $USER apollo_dev_$USER bash -c "./apollo.sh build"

# Step 6: Export Conda environment to ApolloSim/environment.yml
ENV_EXPORT_PATH="$SCRIPT_DIR/../environment.yml"
echo "Exporting Conda environment to $ENV_EXPORT_PATH..."
conda env create -f "$ENV_EXPORT_PATH"

echo "Setup completed successfully!"

## Step 7: Generate Apollo Protos
echo "Generate protoc dependencies"

# Create the output directory if it doesn't exist
mkdir -p "$APOLLO_PROTO_OUT_PATH"

# Find all .proto files, excluding .cache directories
find "$APOLLO_PROTO_PATH" -name '*.proto' | while read -r proto_file; do
    # Check if the file is inside a .cache directory
    if [[ "$proto_file" == *".cache"* ]]; then
        continue
    fi

    # Compile the .proto file
    protoc -I="$APOLLO_PROTO_PATH" --python_out="$APOLLO_PROTO_OUT_PATH" "$proto_file"
done

# generate sample map for apollosim
python $PARENT_DIR/tools/preprocess_ApolloSim_map.py
