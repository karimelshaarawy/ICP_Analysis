#!/usr/bin/env python3
"""
ICP Performance Analysis Script
Analyzes different ICP variants and generates comprehensive reports with visualizations.
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================
import subprocess
import os
import sys
import json
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse

# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================
class ICPAnalyzer:
    """
    Main class responsible for:
    1. Managing ICP configurations
    2. Compiling and running tests
    3. Collecting performance metrics
    4. Generating analysis reports and visualizations
    """
    
    def __init__(self, build_dir="src/build", results_dir="analysis_results"):
        """Initialize analyzer with build and results directories"""
        self.build_dir = Path(build_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load all 16 ICP configurations from external config generator
        from generate_config import get_icp_configs
        self.icp_configs = get_icp_configs()
        
        # Configure for bunny alignment testing (faster than full room reconstruction)
        for config in self.icp_configs:
            config["RUN_SHAPE_ICP"] = 1
            config["RUN_SEQUENCE_ICP"] = 0
        
        # Set testing parameters
        self.num_runs = 3  # Statistical significance through multiple runs
        self.max_iterations = 10  # Reduced for faster testing
        self.test_scenario = "room_reconstruction"
        
    # =========================================================================
    # CONFIGURATION AND COMPILATION METHODS
    # =========================================================================
    
    def update_config(self, config):
        """Generate C++ config.h header file with the given configuration"""
        from generate_config import generate_config
        
        try:
            generate_config(config, "src/config.h")
            print(f"Generated config.h for {config.get('name', 'Unknown')}")
            return True
        except Exception as e:
            print(f"Error generating config: {e}")
            return False
    
    def compile_project(self):
        """Clean and compile the C++ ICP project using make"""
        try:
            # Clean previous build
            result = subprocess.run(
                ["make", "-C", str(self.build_dir), "clean"],
                capture_output=True, text=True
            )
            
            # Compile with new configuration
            result = subprocess.run(
                ["make", "-C", str(self.build_dir)],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                print(f"Compilation failed: {result.stderr}")
                return False
            return True
        except Exception as e:
            print(f"Compilation error: {e}")
            return False
    
    # =========================================================================
    # TEST EXECUTION AND METRICS COLLECTION
    # =========================================================================
    
    def run_icp_test(self, config_name):
        """Execute a single ICP test and capture performance metrics"""
        executable = Path.cwd() / "src" / "build" / "exercise_5"
        
        if not executable.exists():
            print(f"Executable not found: {executable}")
            return None
        
        try:
            # Time the execution and capture output
            start_time = time.time()
            result = subprocess.run(
                [str(executable)],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for complex reconstructions
                cwd=str(self.build_dir)  # Run from build directory where data files are located
            )
            end_time = time.time()
            
            if result.returncode != 0:
                print(f"Execution failed for {config_name}: {result.stderr}")
                print(f"STDOUT: {result.stdout[:500]}...")
                return None
            
            # Extract performance metrics from program output
            print(f"STDOUT length: {len(result.stdout)}")
            print(f"STDOUT preview: {result.stdout[:200]}...")
            metrics = self.parse_metrics(result.stdout, end_time - start_time)
            metrics['config_name'] = config_name
            
            return metrics
            
        except subprocess.TimeoutExpired:
            print(f"Timeout for {config_name}")
            return None
        except Exception as e:
            print(f"Error running {config_name}: {e}")
            return None
    
    def parse_metrics(self, output, total_time):
        """Parse performance metrics from ICP program output"""
        # Initialize metrics dictionary with all possible measurements
        metrics = {
            'total_time': total_time,
            'final_rmse': None,
            'final_mean_distance': None,
            'final_median_distance': None,
            'final_max_distance': None,
            'final_min_distance': None,
            'final_std_deviation': None,
            'final_valid_correspondences': None,
            'final_total_points': None,
            'computation_time': None,
            'correspondence_accuracy_rate': None
        }
        
        # Parse output to find the last (final) metrics section
        lines = output.split('\n')
        
        for line in reversed(lines):
            if "=== ICP Metrics" in line:
                metrics = self.extract_metrics_from_output(lines, metrics)
                break
        
        return metrics
    
    def extract_metrics_from_output(self, lines, metrics):
        """Extract specific numeric values from program output lines"""
        for i, line in enumerate(lines):
            line = line.strip()
            # Parse each metric type using string matching and extraction
            if "RMSE:" in line:
                try:
                    metrics['final_rmse'] = float(line.split("RMSE:")[1].strip())
                except:
                    pass
            elif "Mean Distance:" in line:
                try:
                    metrics['final_mean_distance'] = float(line.split("Mean Distance:")[1].strip())
                except:
                    pass
            elif "Median Distance:" in line:
                try:
                    metrics['final_median_distance'] = float(line.split("Median Distance:")[1].strip())
                except:
                    pass
            elif "Max Distance:" in line:
                try:
                    metrics['final_max_distance'] = float(line.split("Max Distance:")[1].strip())
                except:
                    pass
            elif "Min Distance:" in line:
                try:
                    metrics['final_min_distance'] = float(line.split("Min Distance:")[1].strip())
                except:
                    pass
            elif "Std Deviation:" in line:
                try:
                    metrics['final_std_deviation'] = float(line.split("Std Deviation:")[1].strip())
                except:
                    pass
            elif "Valid Correspondences:" in line:
                try:
                    metrics['final_valid_correspondences'] = int(line.split("Valid Correspondences:")[1].strip())
                except:
                    pass
            elif "Total Points:" in line:
                try:
                    metrics['final_total_points'] = int(line.split("Total Points:")[1].strip())
                except:
                    pass
            elif "Computation Time:" in line:
                try:
                    time_str = line.split("Computation Time:")[1].strip().replace("s", "")
                    metrics['computation_time'] = float(time_str)
                except:
                    pass
        
        # Calculate derived metrics
        if metrics['final_valid_correspondences'] and metrics['final_total_points']:
            metrics['correspondence_accuracy_rate'] = metrics['final_valid_correspondences'] / metrics['final_total_points']
        
        return metrics
    
    # =========================================================================
    # MAIN ANALYSIS ORCHESTRATION
    # =========================================================================
    
    def run_analysis(self):
        """Main method that orchestrates the complete analysis workflow"""
        print("Starting ICP Performance Analysis...")
        print(f"Testing {len(self.icp_configs)} configurations with {self.num_runs} runs each")
        
        all_results = []
        
        # Test each ICP configuration
        for config in self.icp_configs:
            config_name = config['name']
            print(f"\n--- Testing {config_name} ---")
            
            # 1. Generate configuration header file
            self.update_config(config)
            
            # 2. Compile project with new configuration
            if not self.compile_project():
                print(f"Failed to compile for {config_name}, skipping...")
                continue
            
            # 3. Run multiple test iterations for statistical reliability
            config_results = []
            for run in range(self.num_runs):
                print(f"  Run {run + 1}/{self.num_runs}")
                result = self.run_icp_test(config_name)
                if result:
                    result['run'] = run + 1
                    config_results.append(result)
            
            # 4. Calculate average performance across runs
            if config_results:
                avg_result = self.calculate_average_metrics(config_results)
                all_results.append(avg_result)
                
                print(f"  Completed {len(config_results)} runs for {config_name}")
        
        # 5. Save raw results and generate analysis outputs
        self.save_results(all_results)
        self.generate_visualizations(all_results)
        
        print(f"\nAnalysis complete! Results saved to {self.results_dir}")
    
    def calculate_average_metrics(self, config_results):
        """Calculate statistical averages and standard deviations across multiple runs"""
        if not config_results:
            return None
        
        # Single run case - return as-is without std calculations
        if len(config_results) == 1:
            result = config_results[0].copy()
            keys_to_remove = [k for k in result.keys() if k.endswith('_std')]
            for key in keys_to_remove:
                del result[key]
            return result
        
        # Multiple runs - calculate means and standard deviations
        avg_result = config_results[0].copy()
        
        numeric_fields = ['final_rmse', 'final_mean_distance', 'final_median_distance', 
                         'final_max_distance', 'final_min_distance', 'final_std_deviation',
                         'total_time', 'computation_time', 'correspondence_accuracy_rate']
        
        for field in numeric_fields:
            values = [r.get(field) for r in config_results if r.get(field) is not None]
            if values:
                avg_result[field] = np.mean(values)
                avg_result[f'{field}_std'] = np.std(values)
        
        return avg_result
    
    # =========================================================================
    # OUTPUT GENERATION METHODS
    # =========================================================================
    
    def save_results(self, results):
        """Save raw results to timestamped CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = self.results_dir / f"icp_analysis_results_{timestamp}.csv"
        
        with open(csv_file, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        print(f"Results saved to: {csv_file}")
    
    def generate_visualizations(self, results):
        """Generate comprehensive visualization suite showing ICP performance comparisons"""
        if not results:
            print("No results to visualize")
            return
        
        # Set up professional plotting style
        plt.style.use('seaborn-v0_8')
        
        # =====================================================================
        # VISUALIZATION SET 1: Core Performance Metrics
        # =====================================================================
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('ICP Performance Analysis', fontsize=16, fontweight='bold')
        
        self.plot_accuracy_comparison(axes[0], results)  # RMSE comparison
        self.plot_speed_comparison(axes[1], results)     # Computation time comparison
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"icp_analysis_plots_{timestamp}_1.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        # =====================================================================
        # VISUALIZATION SET 2: Advanced Analysis
        # =====================================================================
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('ICP Performance Analysis', fontsize=16, fontweight='bold')
        
        self.plot_correspondence_accuracy(axes[0], results)  # Point correspondence success rate
        self.plot_accuracy_vs_speed(axes[1], results)        # Trade-off analysis
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"icp_analysis_plots_{timestamp}_2.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        # =====================================================================
        # VISUALIZATION SET 3: Specialized Comparisons
        # =====================================================================
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('ICP Performance Analysis', fontsize=16, fontweight='bold')
        
        self.plot_downsampling_impact(axes[0], results)  # Effect of point cloud downsampling
        self.plot_colored_comparison(axes[1], results)   # Colored vs traditional ICP
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"icp_analysis_plots_{timestamp}_3.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_file}")
        
        # Generate textual summary
        self.generate_summary_table(results)
    
    # =========================================================================
    # INDIVIDUAL PLOTTING METHODS
    # =========================================================================
    
    def plot_accuracy_comparison(self, ax, results):
        """Create bar chart comparing final RMSE across all ICP configurations"""
        configs = [r['config_name'] for r in results]
        rmses = [r.get('final_rmse', 0) for r in results]

        # Filter out invalid data points
        valid_data = [(c, r) for c, r in zip(configs, rmses) if r is not None and not np.isnan(r) and r > 0]
        if not valid_data:
            ax.text(0.5, 0.5, 'No valid RMSE data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Final RMSE Comparison (No Data)')
            return

        valid_configs, valid_rmses = zip(*valid_data)
        
        # Use distinct colors for better visual separation
        distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                       '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                       '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                       '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']

        colors = distinct_colors[:len(valid_configs)]

        bars = ax.bar(range(len(valid_configs)), valid_rmses, alpha=0.7, color=colors)
        ax.set_title('Final RMSE Comparison')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('ICP Configuration')
        ax.set_xticks(range(len(valid_configs)))
        ax.set_xticklabels(valid_configs, rotation=45, ha='right')

        # Add value labels on bars
        for bar, rmse in zip(bars, valid_rmses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{rmse:.4f}', ha='center', va='bottom', fontsize=8)

    def plot_speed_comparison(self, ax, results):
        """Create bar chart comparing total computation time across configurations"""
        configs = [r['config_name'] for r in results]
        times = [r.get('total_time', 0) for r in results]

        valid_data = [(c, t) for c, t in zip(configs, times) if t is not None and not np.isnan(t) and t > 0]
        if not valid_data:
            ax.text(0.5, 0.5, 'No valid time data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Computation Time Comparison (No Data)')
            return

        valid_configs, valid_times = zip(*valid_data)

        colors = plt.cm.Set2(np.linspace(0, 1, len(valid_configs)))
        bars = ax.bar(range(len(valid_configs)), valid_times, alpha=0.7, color=colors)
        ax.set_title('Computation Time Comparison')
        ax.set_ylabel('Time (seconds)')
        ax.set_xlabel('ICP Configuration')
        ax.set_xticks(range(len(valid_configs)))
        ax.set_xticklabels(valid_configs, rotation=45, ha='right')

        # Add time labels on bars
        for bar, time_val in zip(bars, valid_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontsize=8)

    def plot_correspondence_accuracy(self, ax, results):
        """Plot percentage of successful point correspondences found by each ICP variant"""
        configs = [r['config_name'] for r in results]
        accuracies = [r.get('correspondence_accuracy_rate', 0) for r in results]
        
        distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                       '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                       '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                       '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']

        colors = distinct_colors[:len(configs)]
        bars = ax.bar(range(len(configs)), accuracies, alpha=0.7, color=colors)
        ax.set_title('Correspondence Accuracy Rate')
        ax.set_ylabel('Accuracy Rate')
        ax.set_xlabel('ICP Configuration')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right')

        # Add percentage labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.2%}', ha='center', va='bottom', fontsize=8)

    def plot_accuracy_vs_speed(self, ax, results):
        """Scatter plot showing the trade-off between accuracy (RMSE) and speed"""
        rmses = [r.get('final_rmse', 0) for r in results]
        times = [r.get('total_time', 0) for r in results]
        configs = [r['config_name'] for r in results]
        
        distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                       '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                       '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                       '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']

        colors = distinct_colors[:len(configs)]
        scatter = ax.scatter(times, rmses, alpha=0.7, s=100, c=colors)
        ax.set_title('Accuracy vs Speed Trade-off')
        ax.set_xlabel('Computation Time (seconds)')
        ax.set_ylabel('RMSE')

        # Add selective labels to avoid clutter
        for i, config in enumerate(configs):
            if i % 3 == 0:  # Label every 3rd point
                ax.annotate(config, (times[i], rmses[i]),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)

    def plot_downsampling_impact(self, ax, results):
        """Analyze how point cloud downsampling affects LinearICP performance"""
        linear_results = [r for r in results if 'LinearICP' in r['config_name']]

        if not linear_results:
            ax.text(0.5, 0.5, 'No LinearICP results', ha='center', va='center', transform=ax.transAxes)
            return

        # Group by downsampling level
        low_results = [r for r in linear_results if 'Low' in r['config_name']]
        medium_results = [r for r in linear_results if 'Medium' in r['config_name']]
        high_results = [r for r in linear_results if 'High' in r['config_name']]

        categories = ['Low', 'Medium', 'High']
        rmses = [
            np.mean([r.get('final_rmse', 0) for r in low_results]) if low_results else 0,
            np.mean([r.get('final_rmse', 0) for r in medium_results]) if medium_results else 0,
            np.mean([r.get('final_rmse', 0) for r in high_results]) if high_results else 0
        ]

        colors = ['#1f77b4', '#ff7f0e', '#d62728']  # Blue, orange, red progression
        bars = ax.bar(categories, rmses, alpha=0.7, color=colors)
        ax.set_title('LinearICP: Downsampling Impact on Accuracy')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Downsampling Level')

        for bar, rmse in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{rmse:.4f}', ha='center', va='bottom')

    def plot_colored_comparison(self, ax, results):
        """Compare performance between color-aware and traditional ICP methods"""
        colored_results = [r for r in results if 'Colored' in r['config_name']]
        non_colored_results = [r for r in results if 'Colored' not in r['config_name']]

        if not colored_results or not non_colored_results:
            ax.text(0.5, 0.5, 'Insufficient data for comparison', ha='center', va='center', transform=ax.transAxes)
            return

        # Calculate averages for each group
        colored_rmse = np.mean([r.get('final_rmse', 0) for r in colored_results])
        non_colored_rmse = np.mean([r.get('final_rmse', 0) for r in non_colored_results])

        colored_time = np.mean([r.get('total_time', 0) for r in colored_results])
        non_colored_time = np.mean([r.get('total_time', 0) for r in non_colored_results])

        categories = ['Non-Colored', 'Colored']
        rmses = [non_colored_rmse, colored_rmse]
        times = [non_colored_time, colored_time]

        # Create dual-axis bar chart
        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax.bar(x - width / 2, rmses, width, label='RMSE', alpha=0.7, color=plt.cm.Paired([0, 2]))
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width / 2, times, width, label='Time (s)', alpha=0.7, color='orange')

        ax.set_title('Colored vs Non-Colored ICP Comparison')
        ax.set_ylabel('RMSE')
        ax2.set_ylabel('Time (seconds)')
        ax.set_xlabel('ICP Type')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)

        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    def generate_summary_table(self, results):
        """Generate comprehensive text-based summary of all results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_file = self.results_dir / f"icp_summary_table_{timestamp}.txt"
        
        with open(table_file, 'w') as f:
            f.write("ICP Performance Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Rank all configurations by accuracy (RMSE)
            sorted_results = sorted(results, key=lambda x: x.get('final_rmse', float('inf')))
            
            f.write("Ranking by Final RMSE (Best to Worst):\n")
            f.write("-" * 40 + "\n")
            for i, result in enumerate(sorted_results, 1):
                config = result['config_name']
                rmse = result.get('final_rmse', 'N/A')
                time_val = result.get('total_time', 'N/A')
                accuracy = result.get('correspondence_accuracy_rate', 'N/A')
                
                f.write(f"{i:2d}. {config:<35} | RMSE: {rmse:8.4f} | Time: {time_val:6.2f}s | Acc: {accuracy:6.2%}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Best Performance by Category:\n")
            f.write("-" * 30 + "\n")
            
            # Find best performer in each ICP family
            categories = {
                'LinearICP': [r for r in results if 'LinearICP' in r['config_name']],
                'LMICP': [r for r in results if 'LMICP' in r['config_name']],
                'SymmetricICP': [r for r in results if 'SymmetricICP' in r['config_name']]
            }
            
            for category, category_results in categories.items():
                if category_results:
                    best = min(category_results, key=lambda x: x.get('final_rmse', float('inf')))
                    f.write(f"{category}: {best['config_name']} (RMSE: {best.get('final_rmse', 'N/A'):.4f})\n")
        
        print(f"Summary table saved to: {table_file}")

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(description='ICP Performance Analysis')
    parser.add_argument('--build-dir', default='src/build', help='Build directory path')
    parser.add_argument('--results-dir', default='analysis_results', help='Results directory path')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per configuration')
    parser.add_argument('--iterations', type=int, default=10, help='Maximum iterations per run')
    
    args = parser.parse_args()
    
    # Initialize and run the complete analysis pipeline
    analyzer = ICPAnalyzer(args.build_dir, args.results_dir)
    analyzer.num_runs = args.runs
    analyzer.max_iterations = args.iterations
    
    analyzer.run_analysis()

if __name__ == "__main__":
    import re
    main() 