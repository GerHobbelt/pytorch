"""
Utility functions for profile analysis.
"""

import hashlib
import logging
import math
import multiprocessing as mp
import os
import pickle
import tempfile
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import Any, Callable, List, Optional, Union

import torch
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils.flop_counter import flop_registry

from .types import _IdxEvt, KernelStats


try:
    import graphviz
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


log = logging.getLogger(__name__)


ATEN_PREFIX = "aten::"


# adapters convert the json trace into a format that works with flops_counter
ArgsType = tuple[tuple[Any, ...], dict[Any, Any]]
AdapterType = Callable[[tuple[Any, ...], tuple[Any, ...]], ArgsType]
adapters_map: dict[str, AdapterType] = {}


def parse_list(lst: str) -> list[int]:
    lst = lst.replace("[", "").replace("]", "")
    substrings = lst.split(",")

    return [int(substring.strip()) for substring in substrings]


def register_adapter(
    aten: Union[str, list[str]],
) -> Callable[
    [AdapterType],
    AdapterType,
]:
    def decorator(func: AdapterType) -> AdapterType:
        global _adapters_map

        if isinstance(aten, str):
            adapters_map[aten] = func
        else:
            for at in aten:
                adapters_map[at] = func
        return func

    return decorator


@register_adapter(["_slow_conv2d_forward"])
def _slow_conv2d_adapter(
    shapes: tuple[Any, ...], concrete: tuple[Any, ...]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)
    tmp.append(False)
    tmp2 = list(concrete)
    if len(tmp2) < 5:
        raise ParseException("slow conv2d has less than 5 concrete inputs")
    tmp2[3] = tmp2[4]
    return conv_adapter(tuple(tmp), tuple(tmp2))


@register_adapter(["convolution", "_convolution", "cudnn_convolution"])
def conv_adapter(
    shapes: tuple[Any, ...], concrete: tuple[Any, ...]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)
    if len(tmp) == 4:
        transposed = False
    elif len(tmp) > 6:
        transposed = bool(tmp[6])
        tmp[6] = transposed
    else:
        raise ParseException(f"Convolution has the wrong number of inputs: {len(tmp)}")

    kwargs: dict[Any, Any] = {}
    if not transposed:
        # calculate output shape if not transposed.
        def conv_out_dims(x: int, kernel: int, stride: int) -> int:
            return (x - kernel) // stride + 1

        stride = parse_list(concrete[3])
        inp = shapes[0]
        w = shapes[1]
        out_x_y = [conv_out_dims(*args) for args in zip(inp[2:], w[2:], stride)]
        out = [inp[0], w[0]] + out_x_y  # we only need the xy values
        kwargs["out_val"] = out

    return tuple(tmp), kwargs


def default_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    return shapes, {}


@register_adapter("addmm")
def addmm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)[:3]
    return tuple(tmp), {}


@register_adapter("bmm")
def bmm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)
    return tuple(tmp[:2]), {}


@register_adapter("baddbmm")
def baddbmm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    tmp = list(shapes)[:3]
    return tuple(tmp), {}


@register_adapter("mm")
def mm_adapter(
    shapes: tuple[Any], concrete: tuple[Any]
) -> tuple[tuple[Any], dict[Any, Any]]:
    return shapes, {}


def _parse_kernel_name(name: str) -> Optional[str]:
    """
    parse the name of the kernel from the event name.
    """
    if name.startswith(ATEN_PREFIX):
        return name[len(ATEN_PREFIX) :]
    elif "conv" in name:
        return "convolution"
    elif "addmm" in name:
        return "addmm"
    elif "bmm" in name:
        return "bmm"
    elif "baddbmm" in name:
        return "baddbmm"
    elif "_mm" in name:
        return "mm"
    else:
        return None


def _calculate_flops(event: dict[str, Any]) -> int:
    """
    This function has to parse the kernel name, which is error prone. There doesn't seem to be another solution that
    will support all the different backends that can generate kernels, so make sure to update this function when new
    ops and backends are desired.
    """
    name = event["name"]
    if "kernel_flop" in event["args"] and event["args"]["kernel_flop"] != 0:
        return event["args"]["kernel_flop"]
    op_name = _parse_kernel_name(name)
    if op_name is None:
        return 0

    op_obj = getattr(torch.ops.aten, op_name, None)
    if op_obj is None or op_obj not in flop_registry:
        return 0

    flop_function = flop_registry[op_obj]

    if "Input Dims" not in event["args"] or "Concrete Inputs" not in event["args"]:
        return 0
    input_shapes = event["args"]["Input Dims"]
    concrete = event["args"]["Concrete Inputs"]
    if op_name in adapters_map:
        try:
            args, kwargs = adapters_map[op_name](input_shapes, concrete)
        except ParseException as e:
            msg = f"Failed to parse {op_name} with {e}"
            log.warning(msg)
            return 0
    else:
        try:
            args, kwargs = default_adapter(input_shapes, concrete)
        except ParseException as e:
            msg = f"Failed to parse {op_name} with {e}"
            log.warning(msg)
            return 0
    return flop_function(*args, **kwargs)


def _get_size_from_string(type_string: str) -> int:
    if not hasattr(torch, type_string):
        return 1
    else:
        return getattr(torch, type_string).itemsize


def _default_estimate_gb(event: dict[str, Any]) -> float:
    sizes_and_types = zip(event["args"]["Input Dims"], event["args"]["Input type"])
    bw = 0
    for size, typ in sizes_and_types:
        isize = _get_size_from_string(typ)
        bw += isize * math.prod(pytree.tree_flatten(size)[0])
    return bw / 1e9


def _estimate_gb(event: dict[str, Any]) -> float:
    """
    Our best effort to estimate the gb, should be refactored soon with MemoryCounter.
    """
    name = event["name"]
    if "kernel_num_gb" in event["args"] and event["args"]["kernel_num_gb"] != 0:
        return event["args"]["kernel_num_gb"]
    if "Input type" not in event["args"] or "Input Dims" not in event["args"]:
        return 0
    op_name = _parse_kernel_name(name)
    if op_name is None:
        return _default_estimate_gb(event)

    op_obj = getattr(torch.ops.aten, op_name, None)
    if op_obj is None:
        return _default_estimate_gb(event)

    if "Input Dims" not in event["args"] or "Concrete Inputs" not in event["args"]:
        return _default_estimate_gb(event)
    input_shapes = event["args"]["Input Dims"]

    # NOTE these will be refactored into a similar object to FlopCounter soon
    def mm_formula(M: int, N: int, K: int, size: int) -> int:
        return 2 * (M * K + N * K + M * N) * size

    if op_name == "addmm":
        add_in_size = math.prod(pytree.tree_flatten(input_shapes[0])[0])
        add_type_size = _get_size_from_string(event["args"]["Input type"][0])
        M = input_shapes[1][0]
        N = input_shapes[1][1]
        assert input_shapes[1][1] == input_shapes[2][0]
        K = input_shapes[2][1]
        mul_type_size = _get_size_from_string(event["args"]["Input type"][1])
        return (mm_formula(M, N, K, mul_type_size) + add_in_size * add_type_size) / 1e9
    elif op_name == "mm":
        M = input_shapes[0][0]
        N = input_shapes[0][1]
        assert input_shapes[0][1] == input_shapes[1][0]
        K = input_shapes[1][1]
        type_size = _get_size_from_string(event["args"]["Input type"][0])
        return mm_formula(M, N, K, type_size) / 1e9
    elif op_name == "baddbmm":
        add_in_size = math.prod(pytree.tree_flatten(input_shapes[0])[0])
        add_type_size = _get_size_from_string(event["args"]["Input type"][0])
        B = input_shapes[0][0]
        M = input_shapes[1][1]
        N = input_shapes[1][2]
        K = input_shapes[2][2]
        mul_type_size = _get_size_from_string(event["args"]["Input type"][1])
        return (
            B * mm_formula(M, N, K, mul_type_size) + add_in_size * add_type_size
        ) / 1e9
    elif op_name == "bmm":
        add_in_size = math.prod(pytree.tree_flatten(input_shapes[0])[0])
        add_type_size = _get_size_from_string(event["args"]["Input type"][0])
        B = input_shapes[0][0]
        M = input_shapes[0][1]
        N = input_shapes[0][2]
        K = input_shapes[1][2]
        mul_type_size = _get_size_from_string(event["args"]["Input type"][1])
        return (
            B * mm_formula(M, N, K, mul_type_size) + add_in_size * add_type_size
        ) / 1e9
    elif op_name in [
        "convolution",
        "_convolution",
        "cudnn_convolution",
        "_slow_conv2d_forward",
    ]:
        concrete = event["args"]["Concrete Inputs"]

        def conv_out_dim(x: int, kernel: int, stride: int) -> int:
            return (x - kernel) // stride + 1

        stride = parse_list(
            concrete[3] if op_name != "_slow_conv2d_forward" else concrete[4]
        )
        inp = input_shapes[0]
        w = input_shapes[1]
        out_x_y = [conv_out_dim(*args) for args in zip(inp[2:], w[2:], stride)]
        out = [inp[0], w[0]] + out_x_y
        # each output element reads in * w * w chunk
        input_reads = out[0] * out[1] * out[2] * out[3] * inp[1] * w[2] * w[3]
        # Assume weights are in cache, so only read once
        weight_reads = w[0] * w[1] * w[2] * w[3]
        return (input_reads + weight_reads) / 1e9

    return _default_estimate_gb(event)


def _create_extern_mapping(
    data: dict[str, Any],
) -> defaultdict[int, list[dict[str, Any]]]:
    """
    compute a mapping from external ids to non kernels, which contain the information we need to estimate flops etc
    """
    extern_mapping: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in data["traceEvents"]:
        if (
            "args" not in event
            or "External id" not in event["args"]
            or event["cat"] != "cpu_op"
        ):
            continue
        if len(extern_mapping[event["args"]["External id"]]) > 0:
            raise ParseException("duplicate external id in event")
        extern_mapping[event["args"]["External id"]].append(event)
    return extern_mapping


def _augment_trace_helper(data: dict[str, Any]) -> dict[str, Any]:
    extern_mapping = _create_extern_mapping(data)

    for event in data["traceEvents"]:
        if "cat" not in event or event["cat"] != "kernel":
            continue
        if "args" not in event:
            raise ParseException(f"kernel has no args: {event}")
        if "External id" not in event["args"]:
            event_str = f"kernel has no External id: {event}"
            log.info(event_str)
            continue

        external_op = extern_mapping[event["args"]["External id"]][0]
        flops = _calculate_flops(external_op)
        if flops == 0:
            flops = _calculate_flops(event)
        external_op["args"]["kernel_flop"] = flops
        external_op["args"]["kernel_num_gb"] = _estimate_gb(external_op)
        event["args"]["kernel_flop"] = external_op["args"]["kernel_flop"]
        event["args"]["kernel_num_gb"] = external_op["args"]["kernel_num_gb"]
    return data


_dtype_map = {
    "float": torch.float,
    "float32": torch.float,
    "int": torch.int,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int,
    "long": torch.long,
    "long int": torch.long,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float64": torch.double,
}


def compute_stats_chunk(args):
    """Compute statistics for a chunk of events."""
    events_chunk, devices, dtype_obj = args

    local_device_stats = {}
    for dev_id, dev in devices.items():
        local_device_stats[dev_id] = defaultdict(OrderedSet)

    for event in events_chunk:
        if "cat" not in event or "args" not in event or event["cat"] != "kernel":
            continue
        if "device" not in event["args"]:
            continue

        dev_tmp = event["args"]["device"]
        if dev_tmp not in devices:
            continue

        dev = devices[dev_tmp]

        dur = event["dur"]  # us
        if "kernel_flop" in event["args"]:
            assert dur != 0
            op_flops = event["args"]["kernel_flop"] / (dur / 1e6)
        else:
            op_flops = 0

        if "kernel_num_gb" in event["args"]:
            assert dur != 0
            op_gbps = event["args"]["kernel_num_gb"] / (dur / 1e6)
        else:
            op_gbps = 0

        if dev.info is not None:
            # Handle dtype properly - it could be a torch.dtype or None
            if (
                dtype_obj is not None
                and hasattr(dev.info, "tops")
                and dtype_obj in dev.info.tops
            ):
                achieved_flops = 100 * op_flops / (1e12 * dev.info.tops[dtype_obj])
            else:
                # Fallback to default dtype or first available
                tops_values = getattr(dev.info, "tops", {})
                if tops_values:
                    # Use the first available dtype's TOPS value
                    first_tops = next(iter(tops_values.values()))
                    achieved_flops = 100 * op_flops / (1e12 * first_tops)
                else:
                    achieved_flops = 0

            achieved_bandwidth = 100 * op_gbps / dev.info.dram_bw_gbs
        else:
            achieved_flops = 0
            achieved_bandwidth = 0

        if "name" not in event:
            continue
        kernel_name = event["name"]
        local_device_stats[dev_tmp][kernel_name].add(
            KernelStats(
                flops=op_flops,
                bw=op_gbps,
                latency=dur,
                achieved_bandwidth=achieved_bandwidth,
                achieved_flops=achieved_flops,
            )
        )

    return local_device_stats


def process_events_chunk(args):
    """Process a chunk of events and return per-tid intervals."""
    events_chunk, start_idx, devices, dtype = args

    per_tid_intervals = defaultdict(list)
    open_stack = defaultdict(list)

    for i, ev in enumerate(events_chunk):
        gi = start_idx + i
        ph = ev.get("ph")
        tid = ev.get("tid", 0)

        if ph == "X":
            ts = ev.get("ts", 0)
            dur = ev.get("dur", 0)
            per_tid_intervals[tid].append(
                _IdxEvt(
                    name=ev.get("name", ""),
                    cat=ev.get("cat", ""),
                    ts=ts,
                    end_ts=ts + dur,
                    tid=tid,
                    parent=None,
                    idx=gi,
                )
            )
        elif ph == "B":
            open_stack[tid].append((ev, gi))
        elif ph == "E":
            if open_stack[tid]:
                beg_ev, beg_idx = open_stack[tid].pop()
                per_tid_intervals[tid].append(
                    _IdxEvt(
                        name=beg_ev.get("name", ""),
                        cat=beg_ev.get("cat", ""),
                        ts=beg_ev.get("ts", 0),
                        end_ts=ev.get("ts", 0),
                        tid=tid,
                        parent=None,
                        idx=beg_idx,
                    )
                )

    return per_tid_intervals, open_stack


# DAG Caching Functions
def get_cache_key(file_path: str) -> str:
    """Generate cache key based on file path, modification time, and size."""
    try:
        stat = os.stat(file_path)
        return hashlib.md5(
            f"{file_path}:{stat.st_mtime}:{stat.st_size}".encode()
        ).hexdigest()
    except OSError:
        # File doesn't exist or can't be accessed
        return hashlib.md5(file_path.encode()).hexdigest()


def get_cache_dir() -> str:
    """Get the cache directory, creating it if necessary."""
    cache_dir = os.path.join(tempfile.gettempdir(), "torch_profile_analysis_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_dag_cached(file_path: str, dtype: str) -> Optional["TraceDAG"]:
    """Load DAG from cache if available and fresh."""
    try:
        cache_dir = get_cache_dir()
        cache_key = get_cache_key(file_path)
        cache_path = os.path.join(cache_dir, f"dag_{cache_key}_{dtype}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                print(f"Loaded DAG from cache for {os.path.basename(file_path)}")
                return cached_data
    except Exception as e:
        print(f"Warning: Failed to load DAG from cache: {e}")
    return None


def save_dag_cached(dag: "TraceDAG", file_path: str, dtype: str):
    """Save DAG to cache."""
    try:
        cache_dir = get_cache_dir()
        cache_key = get_cache_key(file_path)
        cache_path = os.path.join(cache_dir, f"dag_{cache_key}_{dtype}.pkl")

        with open(cache_path, "wb") as f:
            pickle.dump(dag, f)
        print(f"Saved DAG to cache for {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Warning: Failed to save DAG to cache: {e}")


def process_single_trace(args):
    """Process a single trace file and return the DAG. Used for parallel processing."""
    from .json_profile import JsonProfile

    trace_id, input_file, dtype, use_cache = args

    try:
        # Try to load from cache first if caching is enabled
        if use_cache:
            cached_dag = load_dag_cached(input_file, dtype)
            if cached_dag is not None:
                trace_name = os.path.basename(input_file).replace(".json", "")
                return trace_id, cached_dag, trace_name

        profile = JsonProfile(input_file, dtype=dtype)

        # Augment if needed
        if not profile._is_trace_augmented():
            profile.augment_trace()

        dag = profile.build_trace_dag()

        # Save to cache if caching is enabled
        if use_cache:
            save_dag_cached(dag, input_file, dtype)

        # Get trace name from filename
        trace_name = os.path.basename(input_file).replace(".json", "")

        return trace_id, dag, trace_name
    except Exception as e:
        print(f"Error processing trace {trace_id}: {e}")
        raise


def create_multi_trace_visualization(
    input_files: List[str],
    output_file: str,
    dtype: str,
    use_cache: bool = True,
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    format: str = "png",
) -> "MultiTraceDAG":
    """Create a multi-trace DAG visualization from multiple JSON trace files."""
    from .dag_nodes import MultiTraceDAG
    from .json_profile import JsonProfile

    multi_dag = MultiTraceDAG()

    # Use parallel processing for multiple traces
    if use_parallel and len(input_files) > 1:
        return create_multi_trace_visualization_parallel(
            input_files, output_file, dtype, use_cache, max_workers, format
        )

    # Original single-threaded implementation for single trace or when parallel is disabled
    # Load each trace and create individual DAGs
    for trace_id, input_file in enumerate(input_files):
        print(f"Processing trace {trace_id + 1}/{len(input_files)}: {input_file}")

        # Try to load from cache first if caching is enabled
        if use_cache:
            cached_dag = load_dag_cached(input_file, dtype)
            if cached_dag is not None:
                trace_name = os.path.basename(input_file).replace(".json", "")
                multi_dag.add_trace_dag(trace_id, cached_dag, trace_name)
                print(
                    f"  Added {len(cached_dag.nodes)} nodes and {len(cached_dag.edges)} edges from cache"
                )
                continue

        profile = JsonProfile(input_file, dtype=dtype)

        # Augment if needed
        if not profile._is_trace_augmented():
            print(f"  Augmenting trace {trace_id + 1}...")
            profile.augment_trace()

        dag = profile.build_trace_dag()

        # Save to cache if caching is enabled
        if use_cache:
            save_dag_cached(dag, input_file, dtype)

        # Get trace name from filename
        trace_name = os.path.basename(input_file).replace(".json", "")

        # Add to multi-trace DAG
        multi_dag.add_trace_dag(trace_id, dag, trace_name)

        print(
            f"  Added {len(dag.nodes)} nodes and {len(dag.edges)} edges from trace {trace_id + 1}"
        )

    # Assign colors to traces
    multi_dag.assign_colors()

    # Calculate kernel time gradients for color coding
    multi_dag.calculate_kernel_time_gradients()

    # Visualize the multi-trace DAG
    visualize_multi_trace_dag(multi_dag, output_file, format)

    return multi_dag


def create_multi_trace_visualization_parallel(
    input_files: List[str],
    output_file: str,
    dtype: str,
    use_cache: bool = True,
    max_workers: Optional[int] = None,
    format: str = "png",
) -> "MultiTraceDAG":
    """Create a multi-trace DAG visualization using parallel processing."""
    from .dag_nodes import MultiTraceDAG

    multi_dag = MultiTraceDAG()

    if max_workers is None:
        max_workers = min(len(input_files), mp.cpu_count())

    print(
        f"Processing {len(input_files)} traces in parallel using {max_workers} workers..."
    )

    # Prepare arguments for parallel processing
    trace_args = [
        (i, input_file, dtype, use_cache) for i, input_file in enumerate(input_files)
    ]

    # Process traces in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_trace = {
            executor.submit(process_single_trace, args): args for args in trace_args
        }

        for future in as_completed(future_to_trace):
            try:
                trace_id, dag, trace_name = future.result()
                multi_dag.add_trace_dag(trace_id, dag, trace_name)
                print(
                    f"  Completed trace {trace_id + 1}/{len(input_files)}: {len(dag.nodes)} nodes, {len(dag.edges)} edges"
                )
            except Exception as exc:
                args = future_to_trace[future]
                print(f"Trace {args[0]} generated an exception: {exc}")

    # Assign colors and calculate gradients
    multi_dag.assign_colors()
    multi_dag.calculate_kernel_time_gradients()

    # Visualize the multi-trace DAG
    visualize_multi_trace_dag(multi_dag, output_file, format)

    return multi_dag


def visualize_multi_trace_dag(
    multi_dag: "MultiTraceDAG",
    output_path: str = "multi_trace_dag.png",
    format: str = "png",
) -> None:
    """Create a PNG visualization of the multi-trace DAG with composite nodes and colored edges."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Install matplotlib and graphviz.")
        return

    try:
        import graphviz

        dot = graphviz.Digraph(comment="Multi-Trace DAG")
        dot.attr(rankdir="TB")  # Top to bottom layout
        dot.attr("node", shape="box")

        # Create a mapping for safe node names
        safe_names = {}

        # Add nodes with composite design
        for i, (node_name, multi_node) in enumerate(multi_dag.nodes.items()):
            safe_name = f"node_{i}"
            safe_names[node_name] = safe_name

            # Create composite node label
            label = _create_composite_node_label(multi_node, multi_dag)

            # Style based on node type
            if multi_node.node_type == "kernel":
                # Square kernel nodes with composite coloring
                dot.node(
                    safe_name, label, style="filled", fillcolor="white", shape="record"
                )
            else:
                # Rounded operation nodes
                dot.node(
                    safe_name, label, style="filled", fillcolor="white", shape="Mrecord"
                )

        # Group edges by (parent, child) to draw multiple colored edges
        edge_groups = {}
        for parent, child, trace_id in multi_dag.edges:
            key = (parent, child)
            if key not in edge_groups:
                edge_groups[key] = []
            edge_groups[key].append(trace_id)

        # Add edges with trace-specific coloring
        for (parent, child), trace_ids in edge_groups.items():
            if parent in safe_names and child in safe_names:
                parent_safe = safe_names[parent]
                child_safe = safe_names[child]

                # For multiple traces on same edge, create multiple parallel edges
                for i, trace_id in enumerate(trace_ids):
                    color = multi_dag.trace_colors[trace_id]

                    # Add slight offset for multiple edges
                    if len(trace_ids) == 1:
                        dot.edge(parent_safe, child_safe, color=color, penwidth="2")
                    else:
                        # Create slightly different edge styles for multiple traces
                        dot.edge(
                            parent_safe,
                            child_safe,
                            color=color,
                            penwidth="2",
                            constraint="true" if i == 0 else "false",
                        )

        # Add legend
        _add_trace_legend(dot, multi_dag)

        # Render to specified format
        base_path = output_path.rsplit(".", 1)[0] if "." in output_path else output_path
        dot.render(base_path, format=format, cleanup=True)
        print(f"Multi-trace DAG visualization saved to {output_path}")

    except Exception as e:
        print(f"Graphviz multi-trace visualization failed: {e}")
        print("Multi-trace visualization requires graphviz. Please install graphviz.")


def _create_composite_node_label(
    multi_node: "MultiTraceDAGNode", multi_dag: "MultiTraceDAG"
) -> str:
    """Create a composite node label that shows data from each trace."""
    # For single trace, use simpler format
    if len(multi_node.present_in_traces) == 1:
        trace_id = next(iter(multi_node.present_in_traces))
        node = multi_node.trace_instances[trace_id]
        trace_name = multi_dag.trace_names[trace_id]

        if multi_node.node_type == "kernel":
            instance_count = len(node.kernel_instances)
            total_duration = sum(dur for dur, _ in node.kernel_instances)
            # Calculate average duration per instance
            avg_duration = total_duration / instance_count if instance_count > 0 else 0

            # Get gradient color for background
            bg_color = multi_dag.trace_kernel_gradients.get(trace_id, {}).get(
                multi_node.name, "white"
            )
            # For single trace, use regular Graphviz format with \n
            display_name_regular = _wrap_text(multi_node.name, 40)
            display_name_regular = _escape_html(display_name_regular)
            return f"{display_name_regular}|{{<f0> {trace_name}: {instance_count} inst, {total_duration:.1f}μs total, {avg_duration:.1f}μs avg}}"
        else:
            instance_count = getattr(node, "instance_count", 0)
            display_name_regular = _wrap_text(multi_node.name, 40)
            display_name_regular = _escape_html(display_name_regular)
            return f"{display_name_regular}\\n{trace_name}: {instance_count} instances"

    # For multiple traces, create side-by-side layout using HTML table with vertical columns
    sorted_trace_ids = sorted(multi_node.present_in_traces)

    # Create header row with node name spanning all columns
    # Apply escaping and wrapping in a safe way for HTML tables
    safe_name = _safe_html_wrap(multi_node.name, 40)

    num_traces = len(sorted_trace_ids)
    header_row = f'<TR><TD COLSPAN="{num_traces}"><B>{safe_name}</B></TD></TR>'

    # Create data row with side-by-side sections
    data_cells = []
    for trace_id in sorted_trace_ids:
        if trace_id in multi_node.trace_instances:
            node = multi_node.trace_instances[trace_id]
            trace_name = multi_dag.trace_names[trace_id]

            if multi_node.node_type == "kernel":
                # Get gradient color for this trace and kernel
                bg_color = multi_dag.trace_kernel_gradients.get(trace_id, {}).get(
                    multi_node.name, "white"
                )
                instance_count = len(node.kernel_instances)
                total_duration = sum(dur for dur, _ in node.kernel_instances)
                # Calculate average duration per instance
                avg_duration = (
                    total_duration / instance_count if instance_count > 0 else 0
                )

                data_cells.append(
                    f'<TD BGCOLOR="{bg_color}">{trace_name}<BR/>{instance_count} inst<BR/>{total_duration:.1f}μs total<BR/>{avg_duration:.1f}μs avg</TD>'
                )
            else:
                bg_color = "#ffffff"
                instance_count = getattr(node, "instance_count", 0)
                data_cells.append(
                    f'<TD BGCOLOR="{bg_color}">{trace_name}<BR/>{instance_count} instances</TD>'
                )
        else:
            # Empty cell for traces that don't have this node
            bg_color = "#ffffff"
            if multi_node.node_type == "kernel":
                data_cells.append("<TD>-</TD>")
            else:
                data_cells.append(f'<TD BGCOLOR="#{bg_color}">-</TD>')

    data_row = f"<TR>{''.join(data_cells)}</TR>"

    # Combine into HTML table format - Remove CELLBORDER to eliminate double borders
    return f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">{header_row}{data_row}</TABLE>>'


def _safe_html_wrap(text: str, max_width: int) -> str:
    """Safely wrap text for HTML table context by escaping first, then wrapping with HTML breaks."""
    # Step 1: Escape all problematic characters for HTML/XML
    escaped_text = text.replace("&", "&amp;")
    escaped_text = escaped_text.replace("<", "&lt;")
    escaped_text = escaped_text.replace(">", "&gt;")
    escaped_text = escaped_text.replace('"', "&quot;")
    escaped_text = escaped_text.replace("'", "&#39;")
    # Remove other problematic characters
    escaped_text = escaped_text.replace("[", "(")
    escaped_text = escaped_text.replace("]", ")")
    escaped_text = escaped_text.replace("{", "(")
    escaped_text = escaped_text.replace("}", ")")

    # Step 2: Check if wrapping is needed
    if len(escaped_text) <= max_width:
        return escaped_text

    # Step 3: Do HTML-entity-aware wrapping
    lines = []
    current_line = ""
    i = 0

    while i < len(escaped_text):
        # Check if we're at the start of an HTML entity
        if escaped_text[i] == "&":
            # Find the end of the HTML entity
            entity_end = i + 1
            while entity_end < len(escaped_text) and escaped_text[entity_end] != ";":
                entity_end += 1
            if entity_end < len(escaped_text):
                entity_end += 1  # Include the semicolon

            entity = escaped_text[i:entity_end]

            # Check if adding this entity would exceed the line limit
            if len(current_line) + len(entity) > max_width and current_line:
                lines.append(current_line)
                current_line = entity
            else:
                current_line += entity

            i = entity_end
        else:
            # Regular character
            if len(current_line) + 1 > max_width and current_line:
                lines.append(current_line)
                current_line = escaped_text[i]
            else:
                current_line += escaped_text[i]
            i += 1

    if current_line:
        lines.append(current_line)

    return "<BR/>".join(lines)


def _wrap_text(text: str, max_width: int) -> str:
    """Wrap text to fit within max_width characters per line, preserving word boundaries when possible."""
    if len(text) <= max_width:
        return text

    # Split into words for better wrapping
    words = text.split()
    if not words:
        return text

    lines = []
    current_line = words[0]

    for word in words[1:]:
        # Check if adding the next word would exceed the limit
        if len(current_line) + 1 + len(word) <= max_width:
            current_line += " " + word
        else:
            # If current word is too long by itself, split it
            if len(word) > max_width:
                lines.append(current_line)
                # Split the long word across multiple lines
                while len(word) > max_width:
                    lines.append(word[:max_width])
                    word = word[max_width:]
                current_line = word
            else:
                lines.append(current_line)
                current_line = word

    if current_line:
        lines.append(current_line)

    return "\\n".join(lines)


def _wrap_text_html(text: str, max_width: int) -> str:
    """Wrap text to fit within max_width characters per line, using HTML line breaks."""
    if len(text) <= max_width:
        return text

    # Split into words for better wrapping
    words = text.split()
    if not words:
        return text

    lines = []
    current_line = words[0]

    for word in words[1:]:
        # Check if adding the next word would exceed the limit
        if len(current_line) + 1 + len(word) <= max_width:
            current_line += " " + word
        else:
            # If current word is too long by itself, split it
            if len(word) > max_width:
                lines.append(current_line)
                # Split the long word across multiple lines
                while len(word) > max_width:
                    lines.append(word[:max_width])
                    word = word[max_width:]
                current_line = word
            else:
                lines.append(current_line)
                current_line = word

    if current_line:
        lines.append(current_line)

    return "<BR/>".join(lines)


def _escape_html_and_wrap(text: str, max_width: int) -> str:
    """Escape HTML characters then wrap with HTML line breaks, preserving the breaks."""
    # First escape all HTML special characters
    escaped_text = _escape_html(text)

    # Then wrap the escaped text, using a larger effective width since escapes expand the text
    # Estimate: each < or > becomes &lt; or &gt; (4 chars vs 1), so adjust accordingly
    original_chars = text.count("<") + text.count(">")
    escaped_chars = escaped_text.count("&lt;") + escaped_text.count("&gt;")
    expansion_factor = (
        escaped_chars * 3 / max(original_chars, 1) if original_chars > 0 else 1
    )

    # Adjust max_width to account for expansion (be conservative)
    adjusted_width = max(int(max_width / max(expansion_factor, 1.2)), max_width // 2)

    return _wrap_text_html(escaped_text, adjusted_width)


def _escape_html(text: str) -> str:
    """Escape HTML special characters to prevent Graphviz syntax errors."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&#39;")
    # Remove problematic characters that might cause issues
    text = text.replace("[", "(")
    text = text.replace("]", ")")
    text = text.replace("{", "(")
    text = text.replace("}", ")")
    return text


def _add_trace_legend(dot: "graphviz.Digraph", multi_dag: "MultiTraceDAG") -> None:
    """Add a legend showing trace colors."""
    with dot.subgraph(name="cluster_legend") as legend:
        legend.attr(label="Trace Legend", style="filled", fillcolor="lightgray")
        legend.attr("node", shape="plaintext")

        legend_rows = []
        for trace_id in sorted(multi_dag.trace_names.keys()):
            trace_name = multi_dag.trace_names[trace_id]
            color = multi_dag.trace_colors[trace_id]
            legend_rows.append(f'<TR><TD BGCOLOR="{color}">{trace_name}</TD></TR>')

        legend_table = "".join(legend_rows)
        legend.node(
            "legend",
            f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0">{legend_table}</TABLE>>',
        )


class ParseException(RuntimeError):
    pass
