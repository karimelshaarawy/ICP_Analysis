# ICP Performance Analysis Framework

This framework provides comprehensive analysis and comparison of different ICP (Iterative Closest Point) variants implemented in the codebase.

## Overview

The analysis framework systematically tests different ICP configurations and generates:
- **CSV results** with detailed metrics
- **Visualization plots** comparing performance
- **Summary tables** ranking methods by performance
- **Statistical analysis** with confidence intervals

## ICP Variants Tested

### 1. LinearICP
- Point-to-point and point-to-plane constraints
- Different downsampling levels (low/medium/high)
- Colored versions

### 2. LevenbergMarquardtICP
- Point-to-point and point-to-plane constraints
- Colored versions

### 3. SymmetricICP
- Point-to-point and point-to-plane constraints
- Colored versions

## Metrics Collected

### Accuracy Metrics
- **Final RMSE**: Root Mean Square Error of final alignment
- **Mean/Median/Max/Min Distance**: Distance statistics
- **Standard Deviation**: Spread of alignment errors

### Performance Metrics
- **Total Computation Time**: End-to-end processing time
- **Time per Iteration**: Average time per ICP iteration
- **Convergence Rate**: How quickly the method converges

### Correspondence Metrics
- **Valid Correspondence Ratio**: Percentage of valid point matches
- **Total Points**: Number of points processed

## Setup

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure C++ Project is Built
```bash
cd src/build
make clean
make
```

## Usage

### Quick Start
```bash
python analysis_script.py
```

### Advanced Usage
```bash
python analysis_script.py --runs 5 --iterations 15 --build-dir src/build --results-dir my_results
```

### Command Line Options
- `--runs`: Number of runs per configuration (default: 3)
- `--iterations`: Maximum ICP iterations per run (default: 10)
- `--build-dir`: Path to C++ build directory (default: src/build)
- `--results-dir`: Output directory for results (default: analysis_results)

## Output Files

### 1. CSV Results
- `icp_analysis_results_YYYYMMDD_HHMMSS.csv`
- Contains all metrics for each configuration
- Includes statistical measures (mean, std dev)

### 2. Visualization Plots
- `icp_analysis_plots_YYYYMMDD_HHMMSS.png`
- 6 comprehensive plots:
  - Final RMSE comparison
  - Computation time comparison
  - Correspondence accuracy rates
  - Accuracy vs speed trade-off
  - Downsampling impact (LinearICP)
  - Colored vs non-colored comparison

### 3. Summary Table
- `icp_summary_table_YYYYMMDD_HHMMSS.txt`
- Rankings by performance
- Best method per category
- Key insights

## Analysis Workflow

1. **Configuration Setup**: Updates `main.cpp` with each ICP configuration
2. **Compilation**: Recompiles the C++ project for each configuration
3. **Execution**: Runs the ICP executable multiple times per configuration
4. **Metrics Collection**: Parses output to extract performance metrics
5. **Statistical Analysis**: Calculates averages and confidence intervals
6. **Visualization**: Generates comprehensive plots and tables

## Example Results Interpretation

### Best Overall Performance
- **LinearICP_PointToPlane_Medium**: Good balance of speed and accuracy
- **SymmetricICP_PointToPlane**: Best for symmetric scenarios
- **LMICP_PointToPlane**: Robust for large rotations

### Speed vs Accuracy Trade-off
- **Fastest**: LinearICP with high downsampling
- **Most Accurate**: LinearICP with low downsampling and point-to-plane
- **Best Balance**: LinearICP with medium downsampling

### Colored vs Non-Colored
- **Colored ICP**: Better accuracy but slower
- **Non-Colored**: Faster but may be less accurate
- **Recommendation**: Use colored for high-accuracy requirements

## Customization

### Adding New ICP Variants
1. Add new configuration to `get_icp_configs()` in `configure_icp.py`
2. Ensure corresponding flags exist in `main.cpp` (excluding CeresICP)
3. Run analysis to include new variant

### Modifying Metrics
1. Update `parse_metrics()` in `analysis_script.py`
2. Add new visualization functions
3. Update summary table generation

### Changing Test Scenarios
1. Modify the C++ code to use different datasets
2. Update ground truth handling
3. Adjust metric calculations accordingly

## Troubleshooting

### Common Issues

1. **Compilation Errors**
   - Ensure all dependencies are installed
   - Check that `main.cpp` is properly formatted
   - Verify build directory exists

2. **Timeout Errors**
   - Reduce number of iterations (`--iterations 5`)
   - Reduce number of runs (`--runs 2`)
   - Check for infinite loops in ICP implementations

3. **Missing Metrics**
   - Verify that ICP output includes metric logging
   - Check that `calculateAndLogMetrics()` is called
   - Ensure output parsing matches actual format

4. **Poor Performance**
   - Check that downsampling is working correctly
   - Verify that ground truth poses are correct
   - Ensure point clouds are properly aligned

## Performance Tips

1. **For Quick Testing**: Use `--runs 2 --iterations 5`
2. **For Thorough Analysis**: Use `--runs 5 --iterations 15`
3. **For Debugging**: Run individual configurations manually
4. **For Large Datasets**: Use high downsampling levels

## Academic Use

This framework is designed for academic research and provides:
- Reproducible results with statistical significance
- Comprehensive performance comparisons
- Publication-ready visualizations
- Detailed methodology documentation

## Contributing

To add new analysis features:
1. Fork the repository
2. Add new metrics or visualizations
3. Update documentation
4. Submit pull request

## License

This analysis framework is provided as-is for research purposes. 