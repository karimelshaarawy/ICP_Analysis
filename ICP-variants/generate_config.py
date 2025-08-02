#!/usr/bin/env python3
"""
Config Generator for ICP Analysis

This script generates config.h files for different ICP configurations
to enable automated testing of various ICP variants.
"""

import os
import sys
from pathlib import Path

def generate_config(config, output_path="src/config.h"):
    """
    Generate a config.h file with the given configuration
    
    Args:
        config (dict): Configuration dictionary with ICP settings
        output_path (str): Path to output config.h file
    """
    
    config_template = '''#pragma once

/*
 * ICP Optimizer Configuration:
 * 
 * Choose one of the following ICP implementations:
 * - USE_HIERARCHICAL_ICP = 1: Uses HierarchicalICPOptimizer (3-level multi-resolution, fastest)
 * - USE_LINEAR_ICP = 1: Uses LinearICPOptimizer (fast, works well for small rotations)
 * - USE_LM_ICP = 1: Uses LevenbergMarquardtICPOptimizer (robust, handles large rotations)
 * - USE_SYMMETRIC_ICP = 1: Uses AlternatingSymmetricICPOptimizer (symmetric approach)
 * - All = 0: Uses CeresICPOptimizer (default, most robust but requires Ceres)
 * 
 * Constraint Type:
 * - USE_POINT_TO_PLANE = 1: Uses point-to-plane constraints (usually better for surfaces)
 * - USE_POINT_TO_PLANE = 0: Uses point-to-point constraints (faster, works well for point clouds)
 * 
 * Colored ICP:
 * - USE_COLORED_ICP = 1: Uses color information in addition to geometry for better alignment
 * - USE_COLORED_ICP = 0: Uses only geometric information (faster)
 * 
 * Downsampling Level:
 * - DOWNSAMPLING_LEVEL = "low": No downsampling (highest quality, slowest)
 * - DOWNSAMPLING_LEVEL = "medium": 4x downsampling (balanced performance)
 * - DOWNSAMPLING_LEVEL = "high": 8x downsampling (fastest processing)
 * 
 * Test Scenario:
 * - RUN_SHAPE_ICP = 1: Run bunny alignment test (faster, good for development)
 * - RUN_SEQUENCE_ICP = 1: Run room reconstruction test (comprehensive, slower)
 * 
 * Examples:
 * - For bunny alignment with LM-ICP and point-to-plane: USE_LM_ICP=1, USE_POINT_TO_PLANE=1
 * - For room reconstruction with colored symmetric ICP: USE_SYMMETRIC_ICP=1, USE_POINT_TO_PLANE=1, USE_COLORED_ICP=1
 * - For hierarchical ICP with colored constraints: USE_HIERARCHICAL_ICP=1, USE_COLORED_ICP=1
 * - For fast processing: DOWNSAMPLING_LEVEL="high"
 * - For high accuracy: DOWNSAMPLING_LEVEL="low"
 */

#define SHOW_BUNNY_CORRESPONDENCES 0

// ICP Optimizer Selection (only one should be 1)
#define USE_POINT_TO_PLANE {USE_POINT_TO_PLANE}
#define USE_LINEAR_ICP {USE_LINEAR_ICP}
#define USE_LM_ICP {USE_LM_ICP}
#define USE_SYMMETRIC_ICP {USE_SYMMETRIC_ICP}
#define USE_HIERARCHICAL_ICP {USE_HIERARCHICAL_ICP}
#define USE_COLORED_ICP {USE_COLORED_ICP}

// Test Scenario Selection
#define RUN_SHAPE_ICP {RUN_SHAPE_ICP}
#define RUN_SEQUENCE_ICP {RUN_SEQUENCE_ICP}

// Downsampling configuration
// Choose one of: "low", "medium", "high"
// - low: downsampleFactor = 1 (no downsampling, highest quality, slowest)
// - medium: downsampleFactor = 4 (75% reduction, balanced performance)
// - high: downsampleFactor = 8 (87.5% reduction, fastest processing)
#define DOWNSAMPLING_LEVEL "{DOWNSAMPLING_LEVEL}"
'''
    
    # Fill in the configuration values
    config_content = config_template.format(
        USE_POINT_TO_PLANE=config.get('USE_POINT_TO_PLANE', 0),
        USE_LINEAR_ICP=config.get('USE_LINEAR_ICP', 0),
        USE_LM_ICP=config.get('USE_LM_ICP', 0),
        USE_SYMMETRIC_ICP=config.get('USE_SYMMETRIC_ICP', 0),
        USE_HIERARCHICAL_ICP=config.get('USE_HIERARCHICAL_ICP', 0),
        USE_COLORED_ICP=config.get('USE_COLORED_ICP', 0),
        RUN_SHAPE_ICP=config.get('RUN_SHAPE_ICP', 0),
        RUN_SEQUENCE_ICP=config.get('RUN_SEQUENCE_ICP', 1),
        DOWNSAMPLING_LEVEL=config.get('DOWNSAMPLING_LEVEL', 'low')
    )
    
    # Write the config file
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    print(f"Generated config.h for {config.get('name', 'Unknown')}")

def get_icp_configs():
    """
    Get all ICP configurations to test
    
    Returns:
        list: List of configuration dictionaries
    """
    return [
        # LinearICP variants (with different downsampling)
        {"name": "LinearICP_PointToPoint_Low", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 0, "DOWNSAMPLING_LEVEL": "low", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LinearICP_PointToPoint_Medium", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 0, "DOWNSAMPLING_LEVEL": "medium", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LinearICP_PointToPoint_High", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 0, "DOWNSAMPLING_LEVEL": "high", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LinearICP_PointToPlane_Low", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 0, "DOWNSAMPLING_LEVEL": "low", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LinearICP_PointToPlane_Medium", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 0, "DOWNSAMPLING_LEVEL": "medium", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LinearICP_PointToPlane_High", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 0, "DOWNSAMPLING_LEVEL": "high", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LinearICP_Colored_PointToPoint_Low", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 1, "DOWNSAMPLING_LEVEL": "low", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LinearICP_Colored_PointToPlane_Low", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 1, "DOWNSAMPLING_LEVEL": "low", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        
        # LevenbergMarquardtICP variants
        {"name": "LMICP_PointToPoint", "USE_LM_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 0, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LMICP_PointToPlane", "USE_LM_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 0, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LMICP_Colored_PointToPoint", "USE_LM_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 1, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LMICP_Colored_PointToPlane", "USE_LM_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 1, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        
        # SymmetricICP variants
        {"name": "SymmetricICP_PointToPoint", "USE_SYMMETRIC_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 0, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "SymmetricICP_PointToPlane", "USE_SYMMETRIC_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 0, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "SymmetricICP_Colored_PointToPoint", "USE_SYMMETRIC_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 1, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "SymmetricICP_Colored_PointToPlane", "USE_SYMMETRIC_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 1, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
    ]

def main():
    """Main function to generate config files"""
    if len(sys.argv) < 2:
        print("Usage: python3 generate_config.py <config_name>")
        print("Available configs:")
        configs = get_icp_configs()
        for config in configs:
            print(f"  {config['name']}")
        return
    
    config_name = sys.argv[1]
    configs = get_icp_configs()
    
    # Find the requested config
    target_config = None
    for config in configs:
        if config['name'] == config_name:
            target_config = config
            break
    
    if target_config is None:
        print(f"Config '{config_name}' not found!")
        return
    
    # Generate the config file
    generate_config(target_config)
    print(f"Successfully generated config.h for {config_name}")

if __name__ == "__main__":
    main() 