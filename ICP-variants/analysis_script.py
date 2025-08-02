#!/usr/bin/env python3
"""
ICP Performance Analysis Script
Analyzes different ICP variants and generates comprehensive reports with visualizations.
"""

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

class ICPAnalyzer:
    def __init__(self, build_dir="src/build", results_dir="analysis_results"):
        self.build_dir = Path(build_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # ICP configurations to test - all 16 configurations
        from generate_config import get_icp_configs
        self.icp_configs = get_icp_configs()
        
        # Update all configurations to use bunny alignment for faster testing
        for config in self.icp_configs:
            config["RUN_SHAPE_ICP"] = 1
            config["RUN_SEQUENCE_ICP"] = 0
        
        # Analysis parameters
        self.num_runs = 3  # Number of runs per configuration
        self.max_iterations = 10  # Reduced for speed
        self.test_scenario = "room_reconstruction"  # Using room reconstruction
        
    def update_config(self, config):
        """Generate config.h with the given configuration"""
        from generate_config import generate_config
        
        try:
            generate_config(config, "src/config.h")
            print(f"Generated config.h for {config.get('name', 'Unknown')}")
            return True
        except Exception as e:
            print(f"Error generating config: {e}")
            return False
    
    def compile_project(self):
        """Compile the C++ project"""
        try:
            result = subprocess.run(
                ["make", "-C", str(self.build_dir), "clean"],
                capture_output=True, text=True
            )
            
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
    
    def run_icp_test(self, config_name):
        """Run a single ICP test and return metrics"""
        # Use the correct executable path
        executable = Path.cwd() / "src" / "build" / "exercise_5"
        
        if not executable.exists():
            print(f"Executable not found: {executable}")
            return None
        
        try:
            # Run the ICP executable from the build directory where data files are located
            start_time = time.time()
            result = subprocess.run(
                [str(executable)],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for room reconstruction
                cwd=str(self.build_dir)  # Run from build directory
            )
            end_time = time.time()
            
            if result.returncode != 0:
                print(f"Execution failed for {config_name}: {result.stderr}")
                print(f"STDOUT: {result.stdout[:500]}...")  # Print first 500 chars of stdout for debugging
                return None
            
            # Parse metrics from output
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
        """Parse metrics from ICP output"""
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
        
        # Parse the output to extract final metrics
        lines = output.split('\n')
        
        # Look for the last metrics section (any iteration)
        for line in reversed(lines):
            if "=== ICP Metrics" in line:
                # Extract metrics from the following lines
                metrics = self.extract_metrics_from_output(lines, metrics)
                break
        
        return metrics
    
    def extract_metrics_from_output(self, lines, metrics):
        """Extract specific metrics from the output lines"""
        for i, line in enumerate(lines):
            line = line.strip()
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
    
    def run_analysis(self):
        """Run the complete analysis"""
        print("Starting ICP Performance Analysis...")
        print(f"Testing {len(self.icp_configs)} configurations with {self.num_runs} runs each")
        
        all_results = []
        
        for config in self.icp_configs:
            config_name = config['name']
            print(f"\n--- Testing {config_name} ---")
            
            # Generate config.h with this configuration
            self.update_config(config)
            
            # Compile the project
            if not self.compile_project():
                print(f"Failed to compile for {config_name}, skipping...")
                continue
            
            # Run multiple times for statistical significance
            config_results = []
            for run in range(self.num_runs):
                print(f"  Run {run + 1}/{self.num_runs}")
                result = self.run_icp_test(config_name)
                if result:
                    result['run'] = run + 1
                    config_results.append(result)
            
            if config_results:
                # Calculate average metrics for this configuration
                avg_result = self.calculate_average_metrics(config_results)
                all_results.append(avg_result)
                
                print(f"  Completed {len(config_results)} runs for {config_name}")
        
        # Save results
        self.save_results(all_results)
        
        # Generate visualizations
        self.generate_visualizations(all_results)
        
        print(f"\nAnalysis complete! Results saved to {self.results_dir}")
    
    def calculate_average_metrics(self, config_results):
        """Calculate average metrics across multiple runs"""
        if not config_results:
            return None
        
        # For single runs, just return the first result without std fields
        if len(config_results) == 1:
            result = config_results[0].copy()
            # Remove any existing _std fields
            keys_to_remove = [k for k in result.keys() if k.endswith('_std')]
            for key in keys_to_remove:
                del result[key]
            return result
        
        # For multiple runs, calculate averages and standard deviations
        avg_result = config_results[0].copy()
        
        # Calculate averages for numeric metrics
        numeric_fields = ['final_rmse', 'final_mean_distance', 'final_median_distance', 
                         'final_max_distance', 'final_min_distance', 'final_std_deviation',
                         'total_time', 'computation_time', 'correspondence_accuracy_rate']
        
        for field in numeric_fields:
            values = [r.get(field) for r in config_results if r.get(field) is not None]
            if values:
                avg_result[field] = np.mean(values)
                avg_result[f'{field}_std'] = np.std(values)
        
        return avg_result
    
    def save_results(self, results):
        """Save results to CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = self.results_dir / f"icp_analysis_results_{timestamp}.csv"
        
        with open(csv_file, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        print(f"Results saved to: {csv_file}")
    
    def generate_visualizations(self, results):
        """Generate comprehensive visualizations"""
        if not results:
            print("No results to visualize")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ICP Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison (RMSE)
        self.plot_accuracy_comparison(axes[0, 0], results)
        
        # 2. Speed comparison (computation time)
        self.plot_speed_comparison(axes[0, 1], results)
        
        # 3. Correspondence accuracy rate
        self.plot_correspondence_accuracy(axes[0, 2], results)
        
        # 4. Accuracy vs Speed trade-off
        self.plot_accuracy_vs_speed(axes[1, 0], results)
        
        # 5. Downsampling impact (for LinearICP)
        self.plot_downsampling_impact(axes[1, 1], results)
        
        # 6. Colored vs Non-colored comparison
        self.plot_colored_comparison(axes[1, 2], results)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"icp_analysis_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_file}")
        
        # Generate summary table
        self.generate_summary_table(results)
    
    def plot_accuracy_comparison(self, ax, results):
        """Plot RMSE comparison across different ICP methods"""
        configs = [r['config_name'] for r in results]
        rmses = [r.get('final_rmse', 0) for r in results]
        
        # Filter out None values and ensure we have valid data
        valid_data = [(config, rmse) for config, rmse in zip(configs, rmses) 
                      if rmse is not None and not np.isnan(rmse) and rmse > 0]
        
        if not valid_data:
            ax.text(0.5, 0.5, 'No valid RMSE data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Final RMSE Comparison (No Data)')
            return
        
        valid_configs, valid_rmses = zip(*valid_data)
        
        bars = ax.bar(range(len(valid_configs)), valid_rmses, alpha=0.7)
        ax.set_title('Final RMSE Comparison')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('ICP Configuration')
        
        # Rotate x-axis labels for better readability
        ax.set_xticks(range(len(valid_configs)))
        ax.set_xticklabels(valid_configs, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, rmse in zip(bars, valid_rmses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{rmse:.4f}', ha='center', va='bottom', fontsize=8)
    
    def plot_speed_comparison(self, ax, results):
        """Plot computation time comparison"""
        configs = [r['config_name'] for r in results]
        times = [r.get('total_time', 0) for r in results]
        
        # Filter out None values and ensure we have valid data
        valid_data = [(config, time_val) for config, time_val in zip(configs, times) 
                      if time_val is not None and not np.isnan(time_val) and time_val > 0]
        
        if not valid_data:
            ax.text(0.5, 0.5, 'No valid time data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Computation Time Comparison (No Data)')
            return
        
        valid_configs, valid_times = zip(*valid_data)
        
        bars = ax.bar(range(len(valid_configs)), valid_times, alpha=0.7, color='orange')
        ax.set_title('Computation Time Comparison')
        ax.set_ylabel('Time (seconds)')
        ax.set_xlabel('ICP Configuration')
        
        ax.set_xticks(range(len(valid_configs)))
        ax.set_xticklabels(valid_configs, rotation=45, ha='right')
        
        # Add value labels
        for bar, time_val in zip(bars, valid_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{time_val:.2f}s', ha='center', va='bottom', fontsize=8)
    
    def plot_correspondence_accuracy(self, ax, results):
        """Plot correspondence accuracy rate"""
        configs = [r['config_name'] for r in results]
        accuracies = [r.get('correspondence_accuracy_rate', 0) for r in results]
        
        bars = ax.bar(range(len(configs)), accuracies, alpha=0.7, color='green')
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
        """Plot accuracy vs speed trade-off"""
        rmses = [r.get('final_rmse', 0) for r in results]
        times = [r.get('total_time', 0) for r in results]
        configs = [r['config_name'] for r in results]
        
        scatter = ax.scatter(times, rmses, alpha=0.7, s=100)
        ax.set_title('Accuracy vs Speed Trade-off')
        ax.set_xlabel('Computation Time (seconds)')
        ax.set_ylabel('RMSE')
        
        # Add labels for some points
        for i, config in enumerate(configs):
            if i % 3 == 0:  # Label every 3rd point to avoid clutter
                ax.annotate(config, (times[i], rmses[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    def plot_downsampling_impact(self, ax, results):
        """Plot impact of downsampling on LinearICP"""
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
        
        bars = ax.bar(categories, rmses, alpha=0.7, color=['blue', 'orange', 'red'])
        ax.set_title('LinearICP: Downsampling Impact on Accuracy')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Downsampling Level')
        
        # Add value labels
        for bar, rmse in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{rmse:.4f}', ha='center', va='bottom')
    
    def plot_colored_comparison(self, ax, results):
        """Plot colored vs non-colored ICP comparison"""
        # Separate colored and non-colored results
        colored_results = [r for r in results if 'Colored' in r['config_name']]
        non_colored_results = [r for r in results if 'Colored' not in r['config_name']]
        
        if not colored_results or not non_colored_results:
            ax.text(0.5, 0.5, 'Insufficient data for comparison', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calculate average metrics
        colored_rmse = np.mean([r.get('final_rmse', 0) for r in colored_results])
        non_colored_rmse = np.mean([r.get('final_rmse', 0) for r in non_colored_results])
        
        colored_time = np.mean([r.get('total_time', 0) for r in colored_results])
        non_colored_time = np.mean([r.get('total_time', 0) for r in non_colored_results])
        
        categories = ['Non-Colored', 'Colored']
        rmses = [non_colored_rmse, colored_rmse]
        times = [non_colored_time, colored_time]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, rmses, width, label='RMSE', alpha=0.7)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, times, width, label='Time (s)', alpha=0.7, color='orange')
        
        ax.set_title('Colored vs Non-Colored ICP Comparison')
        ax.set_ylabel('RMSE')
        ax2.set_ylabel('Time (seconds)')
        ax.set_xlabel('ICP Type')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    def generate_summary_table(self, results):
        """Generate a summary table of results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_file = self.results_dir / f"icp_summary_table_{timestamp}.txt"
        
        with open(table_file, 'w') as f:
            f.write("ICP Performance Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Sort results by RMSE (best to worst)
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
            
            # Find best in each category
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

def main():
    parser = argparse.ArgumentParser(description='ICP Performance Analysis')
    parser.add_argument('--build-dir', default='src/build', help='Build directory path')
    parser.add_argument('--results-dir', default='analysis_results', help='Results directory path')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per configuration')
    parser.add_argument('--iterations', type=int, default=10, help='Maximum iterations per run')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = ICPAnalyzer(args.build_dir, args.results_dir)
    analyzer.num_runs = args.runs
    analyzer.max_iterations = args.iterations
    
    analyzer.run_analysis()

if __name__ == "__main__":
    import re
    main() 