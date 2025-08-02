#!/usr/bin/env python3
"""
ICP Configuration Helper
Updates main.cpp with different ICP configurations for testing.
"""

import re
import os
from pathlib import Path

def update_main_cpp(config, main_cpp_path="src/main.cpp"):
    """Update main.cpp with the given configuration"""
    
    # Read current main.cpp
    with open(main_cpp_path, 'r') as f:
        content = f.read()
    
    # First, reset all ICP flags to 0
    icp_flags = ["USE_LINEAR_ICP", "USE_LM_ICP", "USE_SYMMETRIC_ICP", "USE_HIERARCHICAL_ICP", "USE_COLORED_ICP", "RUN_SHAPE_ICP", "RUN_SEQUENCE_ICP"]
    for flag in icp_flags:
        pattern = f"#define {flag}\\s+\\d+"
        replacement = f"#define {flag} 0"
        content = re.sub(pattern, replacement, content)
    
    # Update configuration flags
    for flag, value in config.items():
        if flag != "name" and flag != "DOWNSAMPLING_LEVEL":
            # Find and replace the flag definition
            pattern = f"#define {flag}\\s+\\d+"
            replacement = f"#define {flag} {value}"
            content = re.sub(pattern, replacement, content)
    
    # Update downsampling level if specified
    if "DOWNSAMPLING_LEVEL" in config:
        pattern = r'#define DOWNSAMPLING_LEVEL ".*"'
        replacement = f'#define DOWNSAMPLING_LEVEL "{config["DOWNSAMPLING_LEVEL"]}"'
        content = re.sub(pattern, replacement, content)
    
    # Write updated main.cpp
    with open(main_cpp_path, 'w') as f:
        f.write(content)
    
    print(f"Updated main.cpp for configuration: {config.get('name', 'Unknown')}")

def get_icp_configs():
    """Get all ICP configurations to test"""
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

if __name__ == "__main__":
    # Test the configuration update
    configs = get_icp_configs()
    print(f"Total configurations to test: {len(configs)}")
    
    for config in configs:
        print(f"  - {config['name']}")
    
    # Test updating with first config
    if configs:
        update_main_cpp(configs[0]) 