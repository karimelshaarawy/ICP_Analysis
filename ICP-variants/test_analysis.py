#!/usr/bin/env python3
"""
Test script for ICP Analysis Framework
Verifies that the analysis components work correctly.
"""

import subprocess
import sys
import os
from pathlib import Path

def test_python_dependencies():
    """Test if required Python packages are available"""
    required_packages = ['pandas', 'matplotlib', 'seaborn', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✓ All Python dependencies are available")
    return True

def test_cpp_build():
    """Test if C++ project can be built"""
    build_dir = Path("src/build")
    
    if not build_dir.exists():
        print(f"✗ Build directory not found: {build_dir}")
        return False
    
    try:
        # Try to compile
        result = subprocess.run(
            ["make", "-C", str(build_dir), "clean"],
            capture_output=True, text=True
        )
        
        result = subprocess.run(
            ["make", "-C", str(build_dir)],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print(f"✗ Compilation failed: {result.stderr}")
            return False
        
        # Check if executable exists (try both names)
        executable = build_dir / "exercise_5"
        if not executable.exists():
            executable = build_dir / "ICP"
            if not executable.exists():
                print(f"✗ Executable not found: {build_dir}/exercise_5 or {build_dir}/ICP")
                return False
        
        print("✓ C++ project builds successfully")
        return True
        
    except Exception as e:
        print(f"✗ Build test failed: {e}")
        return False

def test_configuration_update():
    """Test if configuration update works"""
    try:
        from configure_icp import update_main_cpp, get_icp_configs
        
        # Test with first configuration
        configs = get_icp_configs()
        if configs:
            test_config = configs[0]
            update_main_cpp(test_config)
            print("✓ Configuration update works")
            return True
        else:
            print("✗ No configurations available")
            return False
            
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_analysis_script():
    """Test if analysis script can be imported"""
    try:
        # Just test import, don't run full analysis
        import analysis_script
        print("✓ Analysis script can be imported")
        return True
    except Exception as e:
        print(f"✗ Analysis script import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing ICP Analysis Framework...\n")
    
    tests = [
        ("Python Dependencies", test_python_dependencies),
        ("C++ Build", test_cpp_build),
        ("Configuration Update", test_configuration_update),
        ("Analysis Script", test_analysis_script),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The analysis framework is ready to use.")
        print("\nTo run the analysis:")
        print("  python analysis_script.py")
    else:
        print("✗ Some tests failed. Please fix the issues before running analysis.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 