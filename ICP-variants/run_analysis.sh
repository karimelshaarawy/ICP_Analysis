#!/bin/bash

# ICP Analysis Runner Script
# This script sets up and runs the ICP performance analysis

echo "=== ICP Performance Analysis Runner ==="
echo

# Check if we're in the right directory
if [ ! -f "src/main.cpp" ]; then
    echo "Error: Please run this script from the project root directory"
    echo "Current directory: $(pwd)"
    echo "Expected files: src/main.cpp"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    echo "Please install Python 3 and try again"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not found"
    echo "Please install pip3 and try again"
    exit 1
fi

echo "1. Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install Python dependencies"
    exit 1
fi

echo "2. Testing the analysis framework..."
python3 test_analysis.py

if [ $? -ne 0 ]; then
    echo "Error: Framework tests failed"
    echo "Please fix the issues before running analysis"
    exit 1
fi

echo "3. Running ICP performance analysis..."
echo "This will test multiple ICP configurations and generate results."
echo "This may take several minutes depending on your system."
echo

# Get user input for analysis parameters
read -p "Number of runs per configuration (default: 3): " runs
runs=${runs:-3}

read -p "Maximum iterations per run (default: 10): " iterations
iterations=${iterations:-10}

read -p "Results directory (default: analysis_results): " results_dir
results_dir=${results_dir:-analysis_results}

echo
echo "Starting analysis with:"
echo "  Runs per configuration: $runs"
echo "  Max iterations per run: $iterations"
echo "  Results directory: $results_dir"
echo

# Run the analysis
python3 analysis_script.py --runs $runs --iterations $iterations --results-dir $results_dir

if [ $? -eq 0 ]; then
    echo
    echo "=== Analysis Complete! ==="
    echo "Results saved to: $results_dir"
    echo
    echo "Generated files:"
    ls -la $results_dir/
    echo
    echo "To view results:"
    echo "  - CSV data: cat $results_dir/icp_analysis_results_*.csv"
    echo "  - Summary: cat $results_dir/icp_summary_table_*.txt"
    echo "  - Plots: open $results_dir/icp_analysis_plots_*.png"
else
    echo "Error: Analysis failed"
    exit 1
fi 