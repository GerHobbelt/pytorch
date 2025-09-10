"""
Profile analysis module for PyTorch Inductor.

This module provides tools for analyzing profiling data from PyTorch operations.
"""

import argparse
import os
import shutil
import sys


# Handle imports for both package usage and direct script execution
try:
    # Try relative imports first (when used as a module)
    from .json_profile import JsonProfile
    from .utils import create_multi_trace_visualization, get_cache_dir
except ImportError:
    # Fall back to absolute imports (when run as a script)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torch._inductor.analysis.json_profile import JsonProfile
    from torch._inductor.analysis.utils import (
        create_multi_trace_visualization,
        get_cache_dir,
    )


def main() -> None:
    """
    Main function for the profile analysis script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diff",
        nargs=5,
        metavar=(
            "input_file1",
            "name1",
            "input_file2",
            "name2",
            "dtype",
        ),
        help="Two json traces to compare with, specified as <file1> <name1> <file2> <name2> <dtype>",
    )
    parser.add_argument(
        "--name_limit",
        type=int,
        help="the maximum name size in the final report",
    )
    parser.add_argument(
        "--augment_trace",
        "-a",
        nargs=3,
        metavar=("input_file", "output_file", "dtype"),
        help="Augment a trace with inductor meta information. Provide input and output file paths.",
    )
    parser.add_argument(
        "--analysis",
        nargs=2,
        metavar=("input_file", "dtype"),
        help="Run analysis on a single trace, specified as <file> <dtype>",
    )
    parser.add_argument(
        "--combine",
        nargs="+",
        metavar=("input_files", "output_file"),
        help="Combine multiple profiles into a single profile by merging trace events. Specify as <input_file1> \
<input_file2> [input_file3 ...] <output_file>. The last argument is the output file, all preceding arguments are \
input files to combine.",
    )
    parser.add_argument(
        "--visualize",
        nargs="+",
        metavar=("input_file", "args"),
        help="Create a DAG visualization of multiple traces showing operation flow from ops to kernels. \
Specify as <input_file1> [input_file2 ...] <output_file> <dtype>. At least 3 arguments required (1 input, output, dtype)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg"],
        default="png",
        help="Output format for visualization (default: png)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing for trace analysis and DAG building (default: True for multiple traces)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing and use single-threaded mode",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of worker processes for parallel processing (default: number of CPU cores)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable DAG caching to disk (default: caching enabled)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the DAG cache directory before processing",
    )
    args = parser.parse_args()

    # Handle cache clearing
    if args.clear_cache:
        cache_dir = get_cache_dir()
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleared cache directory: {cache_dir}")
        else:
            print(f"Cache directory does not exist: {cache_dir}")

    # Determine parallelization settings
    use_parallel = (
        not args.no_parallel
    )  # Default is True unless --no-parallel is specified
    if args.parallel:
        use_parallel = True

    use_cache = not args.no_cache  # Default is True unless --no-cache is specified

    if args.diff:
        p1 = JsonProfile(args.diff[0], args.diff[1], dtype=args.diff[4])
        p1.augment_trace()
        p2 = JsonProfile(args.diff[2], args.diff[3], dtype=args.diff[4])
        p2.augment_trace()
        if args.name_limit:
            print(p1.report(p2, name_limit=args.name_limit))
        else:
            print(p1.report(p2))
    if args.analysis:
        p1 = JsonProfile(
            args.analysis[0],
            dtype=args.analysis[1],
        )
        p1.augment_trace()
        if args.name_limit:
            print(p1.report(name_limit=args.name_limit))
        else:
            print(p1.report())
    if args.augment_trace:
        p = JsonProfile(args.augment_trace[0], dtype=args.augment_trace[2])
        p.augment_trace()
        p.dump(args.augment_trace[1])
    if args.combine:
        input_files = args.combine[:-1]  # All arguments except the last one
        output_file = args.combine[-1]  # Last argument is the output file

        if len(input_files) < 2:
            print("Error: At least 2 input files are required for combining")
            return

        # Load the first profile
        combined = JsonProfile(input_files[0], dtype=None)

        # Iteratively combine with all other profiles
        for input_file in input_files[1:]:
            profile = JsonProfile(input_file, dtype=None)
            combined = combined.combine_with(profile)

        combined.dump(output_file)
        print(f"Successfully combined {', '.join(input_files)} into {output_file}")
    if args.visualize:
        if len(args.visualize) < 3:
            print(
                "Error: --visualize requires at least 3 arguments: <input_file1> <output_file> <dtype>"
            )
            return

        input_files = args.visualize[:-2]  # All but last 2 arguments
        output_file = args.visualize[-2]  # Second to last argument
        dtype = args.visualize[-1]  # Last argument

        print(
            f"Creating multi-trace DAG visualization from {len(input_files)} traces..."
        )
        print(f"Using parallel processing: {use_parallel}")
        print(f"Using caching: {use_cache}")
        print(f"Output format: {args.format}")
        if args.max_workers:
            print(f"Max workers: {args.max_workers}")

        if len(input_files) == 1:
            # Single trace visualization (backward compatibility)
            profile = JsonProfile(input_files[0], dtype=dtype)
            dag = profile.create_trace_dag_visualization(
                output_file, format=args.format
            )
            print(f"DAG visualization completed and saved to {output_file}")
            print(
                f"Found {len(dag.nodes)} nodes and {len(dag.edges)} edges in the trace DAG"
            )
        else:
            # Multi-trace visualization with parallel and caching options
            multi_dag = create_multi_trace_visualization(
                input_files,
                output_file,
                dtype,
                use_cache=use_cache,
                use_parallel=use_parallel,
                max_workers=args.max_workers,
                format=args.format,
            )
            print(f"Multi-trace DAG visualization completed and saved to {output_file}")
            print(
                f"Combined {len(input_files)} traces with {len(multi_dag.nodes)} unique nodes"
            )


if __name__ == "__main__":
    main()
