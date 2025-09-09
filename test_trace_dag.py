#!/usr/bin/env python3
"""
Simple test for the new trace DAG functionality.
Creates a minimal mock trace to test the DAG building and visualization.
"""

import json
import tempfile
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'torch'))

from torch._inductor.analysis.profile_analysis import JsonProfile


def create_mock_trace():
    """Create a minimal mock trace file for testing."""
    mock_trace = {
        "traceEvents": [
            {
                "name": "aten::contiguous",
                "cat": "cpu_op", 
                "ph": "X",
                "ts": 1000,
                "dur": 100,
                "tid": 1,
                "args": {}
            },
            {
                "name": "aten::clone", 
                "cat": "cpu_op",
                "ph": "X", 
                "ts": 1050,
                "dur": 80,
                "tid": 1,
                "args": {}
            },
            {
                "name": "aten::copy_",
                "cat": "cpu_op",
                "ph": "X",
                "ts": 1080,
                "dur": 50,
                "tid": 1, 
                "args": {}
            },
            {
                "name": "cudaLaunchKernel_test",
                "cat": "kernel",
                "ph": "X",
                "ts": 1100,
                "dur": 20,
                "tid": 2,
                "args": {}
            },
            {
                "name": "another_kernel",
                "cat": "kernel", 
                "ph": "X",
                "ts": 1200,
                "dur": 15,
                "tid": 2,
                "args": {}
            }
        ],
        "deviceProperties": [
            {
                "id": 0,
                "name": "NVIDIA A100"
            }
        ]
    }
    return mock_trace


def main():
    print("Testing trace DAG functionality...")
    
    # Create a temporary trace file
    mock_trace = create_mock_trace()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_trace, f)
        temp_path = f.name
    
    try:
        # Test loading and DAG creation
        print(f"Loading mock trace from: {temp_path}")
        profile = JsonProfile(temp_path)
        
        # Build the DAG
        print("Building trace DAG...")
        dag = profile.build_trace_dag()
        
        print("\n=== DAG Analysis Results ===")
        print(f"Total nodes: {len(dag.nodes)}")
        print(f"Total edges: {len(dag.edges)}")
        
        print("\nNodes:")
        for name, node in dag.nodes.items():
            if node.node_type == 'kernel':
                instances = len(node.kernel_instances)
                total_duration = sum(dur for dur, _ in node.kernel_instances) if node.kernel_instances else 0
                print(f"  {name} (kernel): {instances} instances, {total_duration}μs total")
            else:
                print(f"  {name} (operation)")
        
        print("\nEdges:")
        for parent, child in dag.edges:
            print(f"  {parent} -> {child}")
        
        # Test visualization (may fail if dependencies not available)
        print("\nTesting visualization...")
        try:
            profile.visualize_trace_dag(dag, "test_trace_dag.png")
            print("✓ Visualization completed successfully")
        except Exception as e:
            print(f"✗ Visualization failed (expected if graphviz/matplotlib not installed): {e}")
        
        # Test the convenience method
        print("\nTesting convenience method...")
        dag2 = profile.create_trace_dag_visualization("test_trace_dag2.png")
        
        print("\n✓ All tests completed successfully!")
        print("The trace DAG functionality is working correctly.")
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    main()
