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
    
    # Read the current main.cpp file content
    with open(main_cpp_path, 'r') as f:
        content = f.read()
    
    # Reset all ICP algorithm flags to disabled state (0) for clean configuration
    icp_flags = ["USE_LINEAR_ICP", "USE_LM_ICP", "USE_SYMMETRIC_ICP", "USE_HIERARCHICAL_ICP", "USE_COLORED_ICP", "RUN_SHAPE_ICP", "RUN_SEQUENCE_ICP"]
    for flag in icp_flags:
        pattern = f"#define {flag}\\s+\\d+"
        replacement = f"#define {flag} 0"
        content = re.sub(pattern, replacement, content)
    
    # Apply the new configuration by setting specified flags to their desired values
    for flag, value in config.items():
        if flag != "name" and flag != "DOWNSAMPLING_LEVEL":
            # Find and replace the flag definition with new value
            pattern = f"#define {flag}\\s+\\d+"
            replacement = f"#define {flag} {value}"
            content = re.sub(pattern, replacement, content)
    
    # Handle special case for downsampling level (string value rather than numeric)
    if "DOWNSAMPLING_LEVEL" in config:
        pattern = r'#define DOWNSAMPLING_LEVEL ".*"'
        replacement = f'#define DOWNSAMPLING_LEVEL "{config["DOWNSAMPLING_LEVEL"]}"'
        content = re.sub(pattern, replacement, content)
    
    # Write the updated configuration back to main.cpp
    with open(main_cpp_path, 'w') as f:
        f.write(content)
    
    print(f"Updated main.cpp for configuration: {config.get('name', 'Unknown')}")

def get_icp_configs():
    """Get all ICP configurations to test"""
    return [
        # === LinearICP Algorithm Variants ===
        # Test different downsampling levels with point-to-point matching
        {"name": "LinearICP_PointToPoint_Low", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 0, "DOWNSAMPLING_LEVEL": "low", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LinearICP_PointToPoint_Medium", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 0, "DOWNSAMPLING_LEVEL": "medium", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LinearICP_PointToPoint_High", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 0, "DOWNSAMPLING_LEVEL": "high", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        
        # Test different downsampling levels with point-to-plane matching
        {"name": "LinearICP_PointToPlane_Low", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 0, "DOWNSAMPLING_LEVEL": "low", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LinearICP_PointToPlane_Medium", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 0, "DOWNSAMPLING_LEVEL": "medium", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LinearICP_PointToPlane_High", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 0, "DOWNSAMPLING_LEVEL": "high", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        
        # Test colored ICP variants (using color information in matching)
        {"name": "LinearICP_Colored_PointToPoint_Low", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 1, "DOWNSAMPLING_LEVEL": "low", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LinearICP_Colored_PointToPlane_Low", "USE_LINEAR_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 1, "DOWNSAMPLING_LEVEL": "low", "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        
        # === Levenberg-Marquardt ICP Algorithm Variants ===
        # Non-linear optimization approach for ICP registration
        {"name": "LMICP_PointToPoint", "USE_LM_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 0, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LMICP_PointToPlane", "USE_LM_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 0, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LMICP_Colored_PointToPoint", "USE_LM_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 1, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "LMICP_Colored_PointToPlane", "USE_LM_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 1, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        
        # === Symmetric ICP Algorithm Variants ===
        # Bidirectional matching for improved robustness
        {"name": "SymmetricICP_PointToPoint", "USE_SYMMETRIC_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 0, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "SymmetricICP_PointToPlane", "USE_SYMMETRIC_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 0, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "SymmetricICP_Colored_PointToPoint", "USE_SYMMETRIC_ICP": 1, "USE_POINT_TO_PLANE": 0, "USE_COLORED_ICP": 1, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
        {"name": "SymmetricICP_Colored_PointToPlane", "USE_SYMMETRIC_ICP": 1, "USE_POINT_TO_PLANE": 1, "USE_COLORED_ICP": 1, "RUN_SHAPE_ICP": 0, "RUN_SEQUENCE_ICP": 1},
    ]

if __name__ == "__main__":
    # === Main execution for testing configuration system ===
    
    # Get all available ICP configurations for testing
    configs = get_icp_configs()
    print(f"Total configurations to test: {len(configs)}")
    
    # Display all available configuration names
    for config in configs:
        print(f"  - {config['name']}")
    
    # Test the configuration update functionality with the first configuration
    if configs:
        update_main_cpp(configs[0]) 