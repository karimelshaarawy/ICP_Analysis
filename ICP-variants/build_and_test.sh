#!/bin/bash

# Build script for Levenberg-Marquardt ICP implementation

echo "Building ICP variants project..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build the project
make -j4

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    
    # Run the LM ICP test if it exists
    if [ -f "./test_lm_icp" ]; then
        echo "Running LM ICP test..."
        ./test_lm_icp
    else
        echo "Warning: test_lm_icp executable not found"
    fi
    
    # Also run the main exercise if it exists
    if [ -f "./exercise_5" ]; then
        echo "Main exercise executable is available: ./exercise_5"
    fi
else
    echo "Build failed!"
    exit 1
fi

echo "Build and test completed." 