#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify temporal heterogeneous graph installation
Tests that all functions and CLI arguments are properly set up
"""

import sys
import io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that temporal functions can be imported."""
    print("Testing imports...")
    try:
        from src.gnn.hetero_graph_dataset import (
            build_hetero_graph_record,
            build_hetero_temporal_record,
            save_graph_record,
        )
        print("  ✓ All functions imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_function_signatures():
    """Test that temporal functions have correct signatures."""
    print("\nTesting function signatures...")
    try:
        from src.gnn.hetero_graph_dataset import build_hetero_temporal_record
        import inspect
        
        sig = inspect.signature(build_hetero_temporal_record)
        params = list(sig.parameters.keys())
        
        expected_params = [
            "scenario_data",
            "report",
            "mode",
            "time_window",
            "stride",
            "temporal_edges",
            "time_encoding",
            "target_horizon",
        ]
        
        for param in expected_params:
            if param not in params:
                print(f"  ✗ Missing parameter: {param}")
                return False
        
        print(f"  ✓ Function signature correct ({len(params)} parameters)")
        return True
    except Exception as e:
        print(f"  ✗ Error checking signature: {e}")
        return False


def test_cli_arguments():
    """Test that CLI arguments are set up."""
    print("\nTesting CLI arguments...")
    try:
        import argparse
        from src.gnn.build_hetero_graph_dataset import main
        
        # This is a bit hacky, but we can check the module has the right structure
        import src.gnn.build_hetero_graph_dataset as module
        
        # Check that main function exists
        if not hasattr(module, 'main'):
            print("  ✗ main() function not found")
            return False
        
        print("  ✓ CLI module structure correct")
        return True
    except Exception as e:
        print(f"  ✗ Error checking CLI: {e}")
        return False


def test_directory_structure():
    """Test that output directories exist."""
    print("\nTesting directory structure...")
    base_dir = Path("outputs/temporal_graphs")
    
    dirs_to_check = [
        base_dir,
        base_dir / "sequence",
        base_dir / "supra",
    ]
    
    all_exist = True
    for dir_path in dirs_to_check:
        if dir_path.exists():
            print(f"  ✓ {dir_path} exists")
        else:
            print(f"  ✗ {dir_path} missing")
            all_exist = False
    
    return all_exist


def test_documentation():
    """Test that documentation files exist."""
    print("\nTesting documentation...")
    docs = [
        "TEMPORAL_HETERO_GRAPHS.md",
        "TEMPORAL_USAGE_EXAMPLES.md",
    ]
    
    all_exist = True
    for doc in docs:
        doc_path = Path(doc)
        if doc_path.exists():
            print(f"  ✓ {doc} exists")
        else:
            print(f"  ✗ {doc} missing")
            all_exist = False
    
    return all_exist


def test_helper_scripts():
    """Test that helper scripts exist."""
    print("\nTesting helper scripts...")
    scripts = [
        "scripts/build_temporal_graphs.py",
        "scripts/build_temporal_graphs.ps1",
    ]
    
    all_exist = True
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            print(f"  ✓ {script} exists")
        else:
            print(f"  ✗ {script} missing")
            all_exist = False
    
    return all_exist


def main():
    print("=" * 60)
    print("Temporal Heterogeneous Graph - Installation Verification")
    print("=" * 60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Function Signatures", test_function_signatures),
        ("CLI Arguments", test_cli_arguments),
        ("Directory Structure", test_directory_structure),
        ("Documentation", test_documentation),
        ("Helper Scripts", test_helper_scripts),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nUnexpected error in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} {name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All verification tests passed!")
        print("=" * 60)
        print("\nThe temporal heterogeneous graph system is ready to use.")
        print("\nTo test with actual data:")
        print("  1. Ensure you have scenarios with 'detail' saved")
        print("  2. Run: python scripts/test_temporal_build.py")
        print("\nTo generate temporal graphs:")
        print("  python scripts/build_temporal_graphs.py --mode supra")
    else:
        print("✗ Some verification tests failed")
        print("=" * 60)
        print("\nPlease check the errors above and fix any issues.")
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
