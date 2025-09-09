import hashlib
import json
import logging
import math
import mmap
import multiprocessing as mp
import os
import pickle
import tempfile
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch._inductor.analysis.device_info import DeviceSpec, lookup_device_info
from torch._inductor.utils import tabulate_2d, zip_dicts
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils.flop_counter import flop_registry

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


@dataclass
class ProfileEvent:
    category: str
    key: str
    self_device_time_ms: float
    # the benchmark is run multiple times and we average the count across all the
    # runs. It should be an integer but define a float just in case.
    count: float


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


@dataclass(frozen=True)
class KernelStats:
    flops: int
    bw: float
    latency: float  # us
    achieved_flops: float
    achieved_bandwidth: float


KernelNameMap = defaultdict[str, OrderedSet[KernelStats]]


@dataclass(frozen=False)
class Device:
    name: str
    index: int
    info: Optional[DeviceSpec]
    stats: KernelNameMap

    def __repr__(self) -> str:
        return f"Device({self.name}, {self.index}): {self.info}"


@dataclass(frozen=True)
class _IdxEvt:
    name: str
    cat: str
    ts: int
    end_ts: int
    tid: int
    parent: Optional[int]  # index into per-thread array
    idx: int  # global index into self.events


DeviceMap = dict[int, Device]
Table = tuple[list[str], dict[str, list[str]]]


class TraceDAGNode:
    """Represents a node in the DAG - either an operation or a kernel."""

    def __init__(self, name: str, node_type: str):
        self.name = name
        self.node_type = node_type  # 'op' or 'kernel'
        self.kernel_instances: List[Tuple[float, int]] = (
            []
        )  # List of (duration_us, thread_id) for kernels
        self.instance_count: int = (
            0  # Number of times this operation appears in the trace
        )
        # Performance statistics for kernels
        self.achieved_flops_list: List[float] = (
            []
        )  # List of achieved FLOPS % for each instance
        self.achieved_bandwidth_list: List[float] = (
            []
        )  # List of achieved bandwidth % for each instance

        # Multi-trace support
        self.trace_data: Dict[int, Dict] = {}  # Maps trace_id to trace-specific data


class MultiTraceDAGNode:
    """Represents a composite node in the multi-trace DAG."""

    def __init__(self, name: str, node_type: str):
        self.name = name
        self.node_type = node_type  # 'op' or 'kernel'
        self.trace_instances: Dict[int, TraceDAGNode] = {}  # Maps trace_id to node data
        self.present_in_traces: Set[int] = set()  # Which traces contain this node

    def add_trace_instance(self, trace_id: int, node: TraceDAGNode):
        """Add data for this node from a specific trace."""
        self.trace_instances[trace_id] = node
        self.present_in_traces.add(trace_id)


class MultiTraceDAG:
    """Directed Acyclic Graph representing multiple collapsed trace trees."""

    def __init__(self):
        self.nodes: Dict[str, MultiTraceDAGNode] = {}
        self.edges: Set[Tuple[str, str, int]] = (
            set()
        )  # (parent, child, trace_id) relationships
        self.trace_colors: Dict[int, str] = {}
        self.trace_names: Dict[int, str] = {}

    def add_trace_dag(self, trace_id: int, dag: "TraceDAG", trace_name: str):
        """Add a single trace's DAG to the multi-trace DAG."""
        self.trace_names[trace_id] = trace_name

        # Add nodes
        for node_name, node in dag.nodes.items():
            if node_name not in self.nodes:
                self.nodes[node_name] = MultiTraceDAGNode(node_name, node.node_type)
            self.nodes[node_name].add_trace_instance(trace_id, node)

        # Add edges with trace information
        for parent, child in dag.edges:
            self.edges.add((parent, child, trace_id))

    def assign_colors(self):
        """Assign distinct colors to each trace."""
        colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FECA57",
            "#FF9F43",
            "#686DE0",
            "#F8B500",
        ]
        for i, trace_id in enumerate(sorted(self.trace_names.keys())):
            self.trace_colors[trace_id] = colors[i % len(colors)]

    def calculate_kernel_time_gradients(self):
        """Calculate gradient colors based on kernel time percentages for each trace."""
        self.trace_kernel_gradients = {}

        for trace_id in self.trace_names.keys():
            # Calculate total kernel time for this trace
            total_kernel_time = 0.0
            kernel_times = {}

            for node_name, multi_node in self.nodes.items():
                if (
                    multi_node.node_type == "kernel"
                    and trace_id in multi_node.trace_instances
                ):
                    node = multi_node.trace_instances[trace_id]
                    kernel_time = sum(dur for dur, _ in node.kernel_instances)
                    kernel_times[node_name] = kernel_time
                    total_kernel_time += kernel_time

            # Calculate percentages and create gradient mapping
            gradients = {}
            if total_kernel_time > 0:
                max_percentage = (
                    max(kernel_times.values()) / total_kernel_time * 100
                    if kernel_times
                    else 0
                )

                base_color = self.trace_colors[trace_id]

                for node_name, kernel_time in kernel_times.items():
                    percentage = (kernel_time / total_kernel_time) * 100
                    # Create gradient from light to dark based on percentage
                    gradient_color = self._create_gradient_color(
                        base_color, percentage, max_percentage
                    )
                    gradients[node_name] = gradient_color

            self.trace_kernel_gradients[trace_id] = gradients

    def _create_gradient_color(
        self, base_color: str, percentage: float, max_percentage: float
    ) -> str:
        """Create a gradient color from light to dark based on percentage."""
        # Convert hex to RGB
        base_color = base_color.lstrip("#")
        r, g, b = tuple(int(base_color[i : i + 2], 16) for i in (0, 2, 4))

        # Create gradient: 0% = very light, max% = original color
        if max_percentage > 0:
            intensity = percentage / max_percentage
        else:
            intensity = 0

        # Ensure minimum visibility by setting a floor
        min_intensity = 0.2  # Minimum 20% of original color
        max_intensity = 1.0  # 100% of original color

        # Scale intensity between min and max
        scaled_intensity = min_intensity + (max_intensity - min_intensity) * intensity

        # Blend with white: higher intensity = more original color, less white
        final_r = int(255 * (1 - scaled_intensity) + r * scaled_intensity)
        final_g = int(255 * (1 - scaled_intensity) + g * scaled_intensity)
        final_b = int(255 * (1 - scaled_intensity) + b * scaled_intensity)

        return f"#{final_r:02x}{final_g:02x}{final_b:02x}"


class TraceDAG:
    """Directed Acyclic Graph representing the collapsed trace tree."""

    def __init__(self):
        self.nodes: Dict[str, TraceDAGNode] = {}
        self.edges: Set[Tuple[str, str]] = set()  # (parent, child) relationships

    def add_node(self, name: str, node_type: str) -> TraceDAGNode:
        """Add a node to the DAG if it doesn't exist."""
        if name not in self.nodes:
            self.nodes[name] = TraceDAGNode(name=name, node_type=node_type)
        return self.nodes[name]

    def add_edge(self, parent: str, child: str):
        """Add an edge from parent to child."""
        self.edges.add((parent, child))

    def add_kernel_instance(self, kernel_name: str, duration_us: float, thread_id: int):
        """Add a kernel instance to a kernel node."""
        if kernel_name in self.nodes:
            node = self.nodes[kernel_name]
            if node.node_type == "kernel":
                node.kernel_instances.append((duration_us, thread_id))


class JsonProfile:
    _devices: DeviceMap

    def __init__(
        self,
        path: str,
        benchmark_name: Optional[str] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
    ):
        """
        Convenience class for running common operations on chrome/perfetto json traces.
        """
        self.path = path
        with open(path) as f:
            self.data = json.load(f)
            self.events = self.data["traceEvents"]
        self.benchmark_name = benchmark_name
        if dtype is None:
            self.dtype = None
        elif isinstance(dtype, torch.dtype):
            self.dtype = dtype
        else:
            if dtype in _dtype_map:
                self.dtype = _dtype_map[dtype]
            else:
                self.dtype = None
        self._create_devices()

    def convert_dtype(self, event: dict[str, Any]) -> Optional[torch.dtype]:
        """
        Each op has a list of dtypes for each input arg. We need to convert these into a single dtype for flop estimation.
        Issues:
         - converting the strings to concrete torch.dtypes
         - What if we have float32, float, float16 all in the inputs? Our choice is to use the largest buffer dtype.
        """

        if (
            "Input Dims" not in event["args"]
            or "Input type" not in event["args"]
            or "Concrete Inputs" not in event["args"]
        ):
            if "bfloat16" in event["name"]:
                return torch.bfloat16
            elif "float16" in event["name"]:
                return torch.float16
            else:
                return None

        input_sizes = event["args"]["Input Dims"]
        input_types = event["args"]["Input type"]
        concrete_inputs = event["args"]["Concrete Inputs"]
        assert len(input_sizes) == len(input_types)
        assert len(input_types) == len(concrete_inputs)

        if len(input_sizes) == 0:
            raise RuntimeError("Empty input_sizes and input_types")

        biggest_size = 0
        biggest_index = 0
        for i in range(len(input_sizes)):
            if concrete_inputs[i] != "":
                # concrete inputs are usually small tensors, so we can just skip
                continue
            my_size = input_sizes[i]
            total_size = sum(parse_list(my_size))
            if total_size > biggest_size:
                biggest_size = total_size
                biggest_index = i
        ret_type = input_types[biggest_index]
        if ret_type in _dtype_map:
            return _dtype_map[ret_type]
        raise RuntimeError(f"Unknown type: {ret_type}. Please add to _dtype_map.")

    def _create_devices(self) -> None:
        self._devices = {}
        for dev in self.data["deviceProperties"]:
            name = dev["name"]
            device_info = lookup_device_info(name)

            if device_info is None:
                log.info(
                    "Unsupported device in profile: %s, please consider contributing to _device_mapping.",
                    name,
                )
            self._devices[dev["id"]] = Device(
                name, dev["id"], device_info, defaultdict(OrderedSet)
            )

    def calculate_flops(self, event: dict[str, Any]) -> int:
        return _calculate_flops(event)

    def estimate_gb(self, event: dict[str, Any]) -> float:
        return _estimate_gb(event)

    def augment_trace(self) -> None:
        self.data = _augment_trace_helper(self.data)

    def _compute_stats(self, use_parallel: bool = True) -> None:
        """populates the name -> stats map"""
        num_events = len(self.events)

        # Use parallel processing for large traces
        if use_parallel and num_events > 10000:
            self._compute_stats_parallel()
            return

        # Original single-threaded implementation for small traces
        for event in self.events:
            if "cat" not in event or "args" not in event or event["cat"] != "kernel":
                continue
            if "device" not in event["args"]:
                continue
            dev_tmp = event["args"]["device"]
            if dev_tmp not in self._devices:
                continue
            dev = self._devices[event["args"]["device"]]

            dur = event["dur"]  # us
            if "kernel_flop" in event["args"]:
                assert dur != 0
                # 1,000,000us/s * flop / us
                op_flops = event["args"]["kernel_flop"] / (dur / 1e6)
            else:
                op_flops = 0

            if "kernel_num_gb" in event["args"]:
                assert dur != 0
                # 1,000,000us/s * gb  = gb/s
                op_gbps = event["args"]["kernel_num_gb"] / (dur / 1e6)
            else:
                op_gbps = 0

            if dev.info is not None:
                dtype = self.convert_dtype(event) or self.dtype
                if dtype is None:
                    raise RuntimeError(
                        "dtype is not found on tensor and default dtype is not set"
                    )
                achieved_flops = 100 * op_flops / (1e12 * dev.info.tops[dtype])
                achieved_bandwidth = 100 * op_gbps / dev.info.dram_bw_gbs
            else:
                achieved_flops = 0
                achieved_bandwidth = 0

            if "name" not in event:
                continue
            kernel_name = event["name"]
            dev.stats[kernel_name].add(
                KernelStats(
                    flops=op_flops,
                    bw=op_gbps,
                    latency=dur,
                    achieved_bandwidth=achieved_bandwidth,
                    achieved_flops=achieved_flops,
                )
            )

    def _compute_stats_parallel(self) -> None:
        """Parallel version of _compute_stats."""
        num_events = len(self.events)
        num_workers = min(mp.cpu_count(), max(1, num_events // 2000))
        chunk_size = max(1, num_events // num_workers)

        print(
            f"Computing statistics for {num_events} events using {num_workers} workers..."
        )

        # Split events into chunks
        chunks = []
        for i in range(0, num_events, chunk_size):
            chunk = self.events[i : i + chunk_size]
            chunks.append((chunk, self._devices, self.dtype))

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(compute_stats_chunk, chunk_args)
                for chunk_args in chunks
            ]

            # Merge results
            for future in as_completed(futures):
                local_device_stats = future.result()

                # Merge local_device_stats into self._devices
                for dev_id, local_stats in local_device_stats.items():
                    if dev_id in self._devices:
                        dev = self._devices[dev_id]
                        for kernel_name, stats_set in local_stats.items():
                            dev.stats[kernel_name].update(stats_set)

        print("Statistics computation completed.")

    def _create_single_table(self, dev: Device) -> Table:
        """Create a table with the devices mapped to indices."""
        headers = [
            "Kernel Name",
            "Kernel Count",
            "FLOPS",
            "Kernel Reads (GB)",
            "Dur (us)",
            "Achieved FLOPS %",
            "Achieved Bandwidth %",
        ]
        rows: dict[str, list[str]] = {}

        def safe_div_format(x: float, y: float) -> str:
            if y == 0:
                return "0.0"
            return f"{x / y:.4f}"

        for kernel_name, stats_set in dev.stats.items():
            ker_count = 0
            flops = 0
            flops_count = 0
            achieved_flops = 0.0
            bw = 0.0
            bw_count = 0
            achieved_bandwidth = 0.0
            latency = 0.0
            for stats in stats_set:
                if stats.flops != 0:
                    flops += stats.flops
                    achieved_flops += stats.achieved_flops
                    flops_count += 1
                if stats.bw != 0:
                    bw += stats.bw
                    achieved_bandwidth += stats.achieved_bandwidth
                    bw_count += 1
                latency += stats.latency
                ker_count += 1
            assert ker_count != 0
            rows[kernel_name] = [
                str(ker_count),
                safe_div_format(flops, flops_count),
                safe_div_format(bw, bw_count),
                safe_div_format(latency, ker_count),
                safe_div_format(achieved_flops, flops_count),
                safe_div_format(achieved_bandwidth, bw_count),
            ]

        return headers, rows

    def _create_tables(self, devs: DeviceMap) -> dict[int, Table]:
        return {idx: self._create_single_table(dev) for idx, dev in devs.items()}

    def _combine_tables(
        self, table1: Table, table1_name: str, table2: Table, table2_name: str
    ) -> Table:
        new_headers = (
            ["Kernel Name"]
            + [f"{table1_name} {head}" for head in table1[0][1:]]
            + [f"{table2_name} {head}" for head in table2[0][1:]]
        )
        t1_length = len(table1[0][1:])
        t2_length = len(table2[0][1:])
        new_rows = {}

        for key, row1, row2 in zip_dicts(
            table1[1],
            table2[1],
            d1_default=["Empty"] * t1_length,
            d2_default=["Empty"] * t2_length,
        ):
            assert row1 is not None
            assert row2 is not None
            new_rows[key] = row1 + row2
        return new_headers, new_rows

    def report(
        self, other: Optional["JsonProfile"] = None, name_limit: int = 40
    ) -> str:
        def create_ret(
            table_headers: list[str], table_rows: dict[str, list[str]]
        ) -> str:
            table_flattened = [
                [kernel_name[:name_limit], *kernel_vals]
                for kernel_name, kernel_vals in table_rows.items()
            ]
            return tabulate_2d(table_flattened, headers=table_headers)

        if other is not None:
            self._compute_stats()
            other._compute_stats()

            self_tables = self._create_tables(self._devices)
            other_tables = self._create_tables(other._devices)

            self_name = (
                self.benchmark_name if self.benchmark_name is not None else "Table 1"
            )
            other_name = (
                other.benchmark_name if other.benchmark_name is not None else "Table 2"
            )

            ret = []
            assert self._devices.keys() == other._devices.keys()
            for device_idx, t1, t2 in zip_dicts(
                self_tables, other_tables, d1_default=None, d2_default=None
            ):
                assert t1 is not None
                assert t2 is not None
                table_headers, table_rows = self._combine_tables(
                    t1, self_name, t2, other_name
                )
                tab_string = create_ret(table_headers, table_rows)
                ret.append(f"{self._devices[device_idx]}:\n{tab_string}")
            return "\n".join(ret)
        self._compute_stats()

        self_tables = self._create_tables(self._devices)

        ret = []
        for idx, table in self_tables.items():
            table_headers, table_rows = table
            tab_string = create_ret(table_headers, table_rows)
            ret.append(f"{self._devices[idx]}:\n{tab_string}")
        return "\n".join(ret)

    def _build_extern_and_kernel_maps(self, use_parallel: bool = True):
        """Build per-thread intervals with correct parent pointers.
        Also returns kernels and cudaLaunchKernel indices per thread.
        """
        extern_map = _create_extern_mapping(self.data)

        num_events = len(self.events)

        # Use parallel processing for large traces
        if use_parallel and num_events > 5000:
            return self._build_extern_and_kernel_maps_parallel(extern_map)

        # Original single-threaded implementation for small traces
        # 1) Collect intervals per tid (handle 'X' and match 'B'/'E')
        per_tid_intervals: dict[int, list[_IdxEvt]] = defaultdict(list)
        open_stack: dict[int, list[tuple[dict[str, Any], int]]] = defaultdict(list)

        for gi, ev in enumerate(self.events):
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
                        parent=None,  # will fill in below
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
                            parent=None,  # will fill in below
                            idx=beg_idx,
                        )
                    )

        # 2) For each thread, sort by (ts, -end_ts) and rebuild parent pointers via sweep line
        per_tid_compact: dict[int, list[_IdxEvt]] = {}
        launches_per_tid: dict[int, list[_IdxEvt]] = defaultdict(list)
        kernels: list[dict[str, Any]] = []

        for tid, arr in per_tid_intervals.items():
            arr.sort(key=lambda x: (x.ts, -x.end_ts))
            stack: list[int] = []
            rebuilt: list[_IdxEvt] = []

            for ev in arr:
                while stack and rebuilt[stack[-1]].end_ts <= ev.ts:
                    stack.pop()
                parent_idx = stack[-1] if stack else None
                rebuilt.append(
                    _IdxEvt(
                        name=ev.name,
                        cat=ev.cat,
                        ts=ev.ts,
                        end_ts=ev.end_ts,
                        tid=tid,
                        parent=parent_idx,
                        idx=ev.idx,
                    )
                )
                stack.append(len(rebuilt) - 1)

            per_tid_compact[tid] = rebuilt

            # Collect launches and kernels from rebuilt intervals
            for it in rebuilt:
                if "cudaLaunchKer" in it.name:
                    launches_per_tid[tid].append(it)
                if it.cat == "kernel":
                    kernels.append(
                        {
                            "event": self.events[it.idx],
                            "ts": it.ts,
                            "tid": tid,
                            "name": it.name,
                        }
                    )

            # launches need to be time-sorted for bisect
            launches_per_tid[tid].sort(key=lambda x: x.ts)

        return extern_map, kernels, per_tid_compact, launches_per_tid

    def _build_extern_and_kernel_maps_parallel(self, extern_map):
        """Parallel version of _build_extern_and_kernel_maps."""
        num_events = len(self.events)
        num_workers = min(
            mp.cpu_count(), max(1, num_events // 1000)
        )  # At least 1000 events per worker
        chunk_size = max(1, num_events // num_workers)

        print(f"Processing {num_events} events using {num_workers} workers...")

        # Split events into chunks
        chunks = []
        for i in range(0, num_events, chunk_size):
            chunk = self.events[i : i + chunk_size]
            chunks.append((chunk, i, self._devices, self.dtype))

        # Process chunks in parallel using threads (since we're I/O bound on data structures)
        per_tid_intervals_combined = defaultdict(list)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_events_chunk, chunk_args)
                for chunk_args in chunks
            ]

            for future in as_completed(futures):
                per_tid_intervals, _ = future.result()

                # Merge results
                for tid, intervals in per_tid_intervals.items():
                    per_tid_intervals_combined[tid].extend(intervals)

        print(
            f"Event processing completed. Processing {len(per_tid_intervals_combined)} threads..."
        )

        # Continue with the rest of the processing (sorting, parent pointers, etc.)
        # This part is harder to parallelize due to dependencies
        per_tid_compact: dict[int, list[_IdxEvt]] = {}
        launches_per_tid: dict[int, list[_IdxEvt]] = defaultdict(list)
        kernels: list[dict[str, Any]] = []

        # Process each thread's intervals in parallel
        def process_tid_intervals(tid_data):
            tid, arr = tid_data
            arr.sort(key=lambda x: (x.ts, -x.end_ts))

            stack: list[int] = []
            rebuilt: list[_IdxEvt] = []

            for ev in arr:
                while stack and rebuilt[stack[-1]].end_ts <= ev.ts:
                    stack.pop()
                parent_idx = stack[-1] if stack else None
                rebuilt.append(
                    _IdxEvt(
                        name=ev.name,
                        cat=ev.cat,
                        ts=ev.ts,
                        end_ts=ev.end_ts,
                        tid=tid,
                        parent=parent_idx,
                        idx=ev.idx,
                    )
                )
                stack.append(len(rebuilt) - 1)

            # Collect launches and kernels
            launches = []
            tid_kernels = []

            for it in rebuilt:
                if "cudaLaunchKer" in it.name:
                    launches.append(it)
                if it.cat == "kernel":
                    tid_kernels.append(
                        {
                            "event": self.events[it.idx],
                            "ts": it.ts,
                            "tid": tid,
                            "name": it.name,
                        }
                    )

            launches.sort(key=lambda x: x.ts)

            return tid, rebuilt, launches, tid_kernels

        # Process each thread's data in parallel
        with ThreadPoolExecutor(
            max_workers=min(len(per_tid_intervals_combined), num_workers)
        ) as executor:
            futures = [
                executor.submit(process_tid_intervals, (tid, arr))
                for tid, arr in per_tid_intervals_combined.items()
            ]

            for future in as_completed(futures):
                tid, rebuilt, launches, tid_kernels = future.result()
                per_tid_compact[tid] = rebuilt
                launches_per_tid[tid] = launches
                kernels.extend(tid_kernels)

        print(f"Thread processing completed. Found {len(kernels)} kernels.")

        return extern_map, kernels, per_tid_compact, launches_per_tid

    def _find_launch_for_kernel(
        self, kernel_ev: dict, launches_per_tid: dict[int, list[_IdxEvt]]
    ) -> Optional[_IdxEvt]:
        """Find the cudaLaunchKernel that encloses kernel start, using bisect on the kernel's tid and also trying nearby CPU tids when needed."""
        ts_k = kernel_ev.get("ts", 0)
        # Try same tid first (some traces put the launch on the same logical tid as API calls)
        tid = kernel_ev.get("tid", 0)
        for try_tid in (tid,):
            la = launches_per_tid.get(try_tid)
            if not la:
                continue
            # Find rightmost launch with start_ts <= ts_k
            idx = bisect_right([x.ts for x in la], ts_k) - 1
            if idx >= 0:
                cand = la[idx]
                if cand.ts <= ts_k <= cand.end_ts:
                    return cand
        # Fallback: scan a tiny set of other tids (cheap) — prefer the nearest enclosing one by (end_ts - ts_k)
        best = None
        best_slack = 1 << 62
        for la in launches_per_tid.values():
            # binary search to nearest candidate
            idx = bisect_right([x.ts for x in la], ts_k) - 1
            if idx >= 0:
                cand = la[idx]
                if cand.ts <= ts_k <= cand.end_ts:
                    slack = cand.end_ts - ts_k
                    if slack < best_slack:
                        best, best_slack = cand, slack
        return best

    def _collect_chain_from(
        self,
        start_evt: _IdxEvt,
        per_tid_compact: dict[int, list[_IdxEvt]],
        include_all: bool = True,
    ) -> list[str]:
        """Walk to the root, returning names from outermost → leaf.
        include_all=True keeps user annotations like 'expected' so you see the full tree.
        """
        arr = per_tid_compact[start_evt.tid]
        chain: list[str] = []
        cur = start_evt
        tmp: list[str] = []
        while cur is not None:
            nm = cur.name
            if include_all or (
                nm.startswith("aten::")
                or "contiguous" in nm
                or "clone" in nm
                or "copy" in nm
                or "empty" in nm
                or nm.startswith("torch::")
                or nm.startswith("c10::")
                or any(
                    op in nm
                    for op in ("linear", "conv", "matmul", "bmm", "addmm", "mm")
                )
            ):
                tmp.append(nm)
            cur = arr[cur.parent] if (cur.parent is not None) else None
        tmp.reverse()
        chain.extend(tmp)
        return chain

    def build_trace_dag(self) -> "TraceDAG":
        """
        Fast DAG build:
        - Pre-index per-thread with parent pointers (O(N))
        - Resolve each kernel to a cpu site (External id or cudaLaunch via bisect) (O(log N) each)
        - Add edges only (set handles de-dupe)
        The slow bits are from (a) O(K·N) overlap scans and (b) de-duping whole chains. Below is a drop-in, sweep-line + parent-pointer approach that makes everything essentially O(N log N):
        Pre-index every cpu_op / user_annotation / cudaLaunchKernel by thread, with parent pointers built from the per-thread stack (no overlap scans).
        For each kernel, resolve its launching site fast:
        Prefer args["External id"] → cpu_op (your existing mapping).
        Else, find the cudaLaunchKernel whose interval contains the kernel start using bisect over a per-thread sorted list (O(log N)).
        Build the op chain by walking parent pointers from the launch (or external op) up to the root; don't de-dup chains—just add edges (the set takes care of uniqueness).
        Intern strings and store compact structs to keep memory/cache friendly.
        Notes on why this is fast
        No per-kernel O(N) overlap scans; ancestor resolution is O(log N) via bisect → O(K log N).
        Parent pointers are built in one linear sweep per thread.
        No chain de-duplication pass; sets make edge/node insertion idempotent.
        """
        dag = TraceDAG()

        # Compute stats first to have performance data available
        self._compute_stats()

        extern_map, kernels, per_tid_compact, launches_per_tid = (
            self._build_extern_and_kernel_maps()
        )

        # Track operation instance counts
        op_instance_counts = defaultdict(int)

        # Pre-intern node objects to reduce dict churn
        def _get_or_add(name: str, typ: str):
            node = dag.nodes.get(name)
            if node is None:
                node = dag.add_node(name, typ)
            return node

        for k in kernels:
            kev = k["event"]
            kname = kev.get("name", "unknown_kernel")
            kdur = kev.get("dur", 0.0)
            ktid = kev.get("tid", 0)

            # 1) Prefer External id mapping to a concrete cpu_op
            start_chain_names: list[str] = []
            ext_ok = False
            if "args" in kev and "External id" in kev["args"]:
                ext_id = kev["args"]["External id"]
                lst = extern_map.get(ext_id)
                if lst:
                    # Use the cpu_op we mapped; find its compact record via per-thread binary search
                    cpu_ev = lst[0]
                    tid = cpu_ev.get("tid", 0)
                    arr = per_tid_compact.get(tid, [])
                    if arr:
                        # binary search nearest exact match by ts
                        ts = cpu_ev.get("ts", 0)
                        idx = bisect_right([x.ts for x in arr], ts) - 1
                        # walk forward to the first with same ts/name if needed
                        found = None
                        for j in range(max(idx, 0), min(idx + 4, len(arr))):
                            if arr[j].ts == ts and arr[j].name == cpu_ev.get(
                                "name", ""
                            ):
                                found = arr[j]
                                break
                        if found:
                            start_chain_names = self._collect_chain_from(
                                found, per_tid_compact
                            )
                            ext_ok = True

            # 2) Else resolve launch site by bisect
            if not ext_ok:
                launch = self._find_launch_for_kernel(kev, launches_per_tid)
                if launch:
                    start_chain_names = self._collect_chain_from(
                        launch, per_tid_compact
                    )

            # If nothing found, skip — we only keep kernel-linked chains
            if not start_chain_names:
                continue

            # Count operation instances and add op nodes and edges
            for nm in start_chain_names:
                op_instance_counts[nm] += 1

            prev = None
            for nm in start_chain_names:
                _get_or_add(nm, "op")
                if prev is not None:
                    dag.add_edge(prev, nm)
                prev = nm

            # Add the kernel node + edge from last op
            kernel_node = _get_or_add(kname, "kernel")
            if prev is not None:
                dag.add_edge(prev, kname)
            dag.add_kernel_instance(kname, float(kdur), int(ktid))

            # Collect performance statistics for this kernel instance
            if "device" in kev.get("args", {}):
                device_id = kev["args"]["device"]
                if device_id in self._devices:
                    dev = self._devices[device_id]
                    if kname in dev.stats:
                        # For each kernel instance, add its performance stats
                        # Since the stats are collected from the same kernel events we're processing,
                        # we can use all stats for this kernel name
                        stats_list = list(dev.stats[kname])
                        if stats_list:
                            # Find the best matching stats by latency
                            best_stats = min(
                                stats_list, key=lambda s: abs(s.latency - kdur)
                            )
                            latency_diff = abs(best_stats.latency - kdur)
                            if (
                                latency_diff < 100.0
                            ):  # Increase tolerance to 100 microseconds
                                kernel_node.achieved_flops_list.append(
                                    best_stats.achieved_flops
                                )
                                kernel_node.achieved_bandwidth_list.append(
                                    best_stats.achieved_bandwidth
                                )

        # Store operation instance counts in the DAG nodes
        for op_name, count in op_instance_counts.items():
            if op_name in dag.nodes:
                node = dag.nodes[op_name]
                if node.node_type == "op":
                    # Add instance count to operation nodes
                    node.instance_count = count

        return dag

    def _trace_up_from_kernel(
        self, kernel_info: Dict, thread_stacks: Dict
    ) -> List[Dict]:
        """
        Trace up from a kernel to find the operation chain that led to it.
        This uses timing overlap to find parent operations.
        """
        chain = []
        kernel_event = kernel_info["event"]
        kernel_ts = kernel_info["ts"]
        kernel_dur = kernel_info.get("dur", 0)
        kernel_end_ts = kernel_ts + kernel_dur

        # Find events that overlap with the kernel timing
        overlapping_ops = []

        for event in self.events:
            if event.get("cat") not in ["cpu_op", "user_annotation"]:
                continue

            event_ts = event.get("ts", 0)
            event_dur = event.get("dur", 0)
            event_end_ts = event_ts + event_dur

            # Check if events overlap
            if event_ts <= kernel_end_ts and event_end_ts >= kernel_ts:
                op_name = event.get("name", "")

                # Filter for relevant operations (aten:: ops)
                if op_name.startswith("aten::") or any(
                    x in op_name for x in ["contiguous", "clone", "copy", "empty"]
                ):
                    overlapping_ops.append(
                        {
                            "name": op_name,
                            "ts": event_ts,
                            "dur": event_dur,
                            "end_ts": event_end_ts,
                            "event": event,
                        }
                    )

        # Sort by start time to build the chain
        overlapping_ops.sort(key=lambda x: x["ts"])

        # Build the operation chain
        for op in overlapping_ops:
            chain.append(op)

        # Add the kernel at the end
        chain.append(
            {
                "name": kernel_info["name"],
                "ts": kernel_ts,
                "dur": kernel_dur,
                "event": kernel_event,
            }
        )

        return chain

    def visualize_trace_dag(
        self, dag: TraceDAG, output_path: str = "trace_dag.png"
    ) -> None:
        """
        Create a PNG visualization of the trace DAG with operations at top and kernels at bottom.
        """
        if not VISUALIZATION_AVAILABLE:
            print(
                "Visualization libraries not available. Install matplotlib and graphviz."
            )
            return

        # Use graphviz for clean DAG layout
        try:
            import graphviz

            dot = graphviz.Digraph(comment="Trace DAG")
            dot.attr(rankdir="TB")  # Top to bottom layout
            dot.attr("node", shape="box")

            # Add nodes with different styles for ops vs kernels
            # Create a mapping for safe node names (graphviz doesn't like special chars)
            safe_names = {}
            for i, (node_name, node) in enumerate(dag.nodes.items()):
                # Create safe name for graphviz
                safe_name = f"node_{i}"
                safe_names[node_name] = safe_name

                if node.node_type == "kernel":
                    # Kernel nodes at bottom with instance count and performance stats
                    instance_count = len(node.kernel_instances)
                    total_duration = sum(dur for dur, _ in node.kernel_instances)

                    # Truncate long kernel names for display
                    display_name = (
                        node_name[:50] + "..." if len(node_name) > 50 else node_name
                    )

                    label = f"{display_name}\\n{instance_count} instances\\n{total_duration:.2f}μs total"

                    # Add performance statistics if available and non-zero
                    if node.achieved_flops_list:
                        flops_min = min(node.achieved_flops_list)
                        flops_max = max(node.achieved_flops_list)
                        flops_avg = sum(node.achieved_flops_list) / len(
                            node.achieved_flops_list
                        )
                        # Only show FLOPS stats if they're not all zero
                        if flops_max > 0.0:
                            label += f"\\nFLOPS %: min={flops_min:.1f}, max={flops_max:.1f}, avg={flops_avg:.1f}"

                    if node.achieved_bandwidth_list:
                        bw_min = min(node.achieved_bandwidth_list)
                        bw_max = max(node.achieved_bandwidth_list)
                        bw_avg = sum(node.achieved_bandwidth_list) / len(
                            node.achieved_bandwidth_list
                        )
                        # Only show BW stats if they're not all zero
                        if bw_max > 0.0:
                            label += f"\\nBW %: min={bw_min:.1f}, max={bw_max:.1f}, avg={bw_avg:.1f}"

                    dot.node(safe_name, label, style="filled", fillcolor="lightcoral")
                else:
                    # Operation nodes with instance counts
                    instance_count = getattr(node, "instance_count", 0)
                    # Truncate long operation names for display
                    display_name = (
                        node_name[:50] + "..." if len(node_name) > 50 else node_name
                    )
                    label = f"{display_name}"
                    if instance_count > 0:
                        label += f"\\n{instance_count} instances"
                    dot.node(safe_name, label, style="filled", fillcolor="lightblue")

            # Add edges using safe names
            for parent, child in dag.edges:
                if parent in safe_names and child in safe_names:
                    dot.edge(safe_names[parent], safe_names[child])

            # Render to PNG
            dot.render(output_path.replace(".png", ""), format="png", cleanup=True)
            print(f"DAG visualization saved to {output_path}")

        except Exception as e:
            print(f"Graphviz visualization failed: {e}")
            # Fallback to matplotlib
            self._visualize_dag_matplotlib(dag, output_path)

    def _visualize_dag_matplotlib(self, dag: TraceDAG, output_path: str) -> None:
        """Fallback visualization using matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Simple layout: ops at top, kernels at bottom
        op_nodes = [name for name, node in dag.nodes.items() if node.node_type == "op"]
        kernel_nodes = [
            name for name, node in dag.nodes.items() if node.node_type == "kernel"
        ]

        # Position nodes
        positions = {}

        # Ops at top
        for i, name in enumerate(op_nodes):
            positions[name] = (i * 2, 1)

        # Kernels at bottom
        for i, name in enumerate(kernel_nodes):
            positions[name] = (i * 2, 0)

        # Draw nodes
        for name, (x, y) in positions.items():
            node = dag.nodes[name]
            if node.node_type == "kernel":
                color = "lightcoral"
                instance_count = len(node.kernel_instances)
                label = f"{name}\n({instance_count})"
            else:
                color = "lightblue"
                label = name

            rect = FancyBboxPatch(
                (x - 0.4, y - 0.2),
                0.8,
                0.4,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="black",
            )
            ax.add_patch(rect)
            ax.text(x, y, label, ha="center", va="center", fontsize=8)

        # Draw edges
        for parent, child in dag.edges:
            if parent in positions and child in positions:
                x1, y1 = positions[parent]
                x2, y2 = positions[child]
                ax.arrow(
                    x1,
                    y1 - 0.2,
                    x2 - x1,
                    y2 - y1 + 0.4,
                    head_width=0.05,
                    head_length=0.05,
                    fc="black",
                    ec="black",
                )

        ax.set_xlim(-1, max(len(op_nodes), len(kernel_nodes)) * 2)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Trace Operation DAG")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"DAG visualization saved to {output_path}")

    def _is_trace_augmented(self) -> bool:
        """Check if the trace has been augmented with performance information."""
        for event in self.events:
            if event.get("cat") == "kernel" and "args" in event:
                # If we find a kernel with performance data, the trace is augmented
                if "kernel_flop" in event["args"] or "kernel_num_gb" in event["args"]:
                    return True
        return False

    def create_trace_dag_visualization(
        self, output_path: str = "trace_dag.png"
    ) -> TraceDAG:
        """
        Convenience method to build and visualize the trace DAG in one step.
        Automatically augments the trace if it hasn't been augmented yet.
        Returns the created DAG for further analysis.
        """
        # Check if trace needs augmentation
        if not self._is_trace_augmented():
            print("Trace not augmented with performance data. Augmenting trace...")
            self.augment_trace()
            print("Trace augmentation completed.")
        else:
            print("Trace already augmented with performance data.")

        print("Building trace DAG from JSON profile...")
        dag = self.build_trace_dag()

        print(f"DAG created with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
        print(
            f"Operations: {[name for name, node in dag.nodes.items() if node.node_type == 'op']}"
        )
        print(
            f"Kernels: {[name for name, node in dag.nodes.items() if node.node_type == 'kernel']}"
        )

        # Check if performance statistics are available
        kernels_with_perf_stats = sum(
            1
            for node in dag.nodes.values()
            if node.node_type == "kernel"
            and (
                (node.achieved_flops_list and max(node.achieved_flops_list) > 0)
                or (
                    node.achieved_bandwidth_list
                    and max(node.achieved_bandwidth_list) > 0
                )
            )
        )

        if kernels_with_perf_stats == 0:
            print(
                """
                Note: No performance statistics (FLOPS/bandwidth %) are displayed because the trace lacks tensor \
shape and type information needed for calculations. To include performance metrics, run torch.profiler.profile with record_shapes=True
                """
            )

        print(f"Visualizing DAG to {output_path}...")
        self.visualize_trace_dag(dag, output_path)

        return dag

    def dump(self, out: str) -> None:
        with open(out, "w") as f:
            json.dump(self.data, f)

    def combine_with(self, other: "JsonProfile") -> "JsonProfile":
        """
        Combine this profile with another profile by merging their trace events.
        Returns a new JsonProfile object with combined data.
        """
        # Create a new combined data structure
        combined_data = {
            "traceEvents": self.data["traceEvents"] + other.data["traceEvents"],
            "deviceProperties": self.data.get("deviceProperties", []),
        }

        # Merge device properties, avoiding duplicates
        other_device_props = other.data.get("deviceProperties", [])
        existing_device_ids = OrderedSet(
            [dev["id"] for dev in combined_data["deviceProperties"]]
        )

        for device_prop in other_device_props:
            if device_prop["id"] not in existing_device_ids:
                combined_data["deviceProperties"].append(device_prop)

        # Copy any other top-level properties from the first profile
        for key, value in self.data.items():
            if key not in combined_data:
                combined_data[key] = value

        import os

        # Create a temporary file to write the combined data
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(combined_data, tmp_file)
            tmp_path = tmp_file.name

        try:
            # Create new JsonProfile from the combined data
            combined_profile = JsonProfile(
                tmp_path,
                benchmark_name=f"{self.benchmark_name or 'Profile1'}_+_{other.benchmark_name or 'Profile2'}",
                dtype=self.dtype or other.dtype,
            )
            return combined_profile
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)


class ParseException(RuntimeError):
    pass


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


def load_dag_cached(file_path: str, dtype: str) -> Optional[TraceDAG]:
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


def save_dag_cached(dag: TraceDAG, file_path: str, dtype: str):
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


def create_multi_trace_visualization(
    input_files: List[str],
    output_png: str,
    dtype: str,
    use_cache: bool = True,
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
) -> MultiTraceDAG:
    """Create a multi-trace DAG visualization from multiple JSON trace files."""
    multi_dag = MultiTraceDAG()

    # Use parallel processing for multiple traces
    if use_parallel and len(input_files) > 1:
        return create_multi_trace_visualization_parallel(
            input_files, output_png, dtype, use_cache, max_workers
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
    visualize_multi_trace_dag(multi_dag, output_png)

    return multi_dag


def create_multi_trace_visualization_parallel(
    input_files: List[str],
    output_png: str,
    dtype: str,
    use_cache: bool = True,
    max_workers: Optional[int] = None,
) -> MultiTraceDAG:
    """Create a multi-trace DAG visualization using parallel processing."""
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
    visualize_multi_trace_dag(multi_dag, output_png)

    return multi_dag


def visualize_multi_trace_dag(
    multi_dag: MultiTraceDAG, output_path: str = "multi_trace_dag.png"
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

        # Render to PNG
        dot.render(output_path.replace(".png", ""), format="png", cleanup=True)
        print(f"Multi-trace DAG visualization saved to {output_path}")

    except Exception as e:
        print(f"Graphviz multi-trace visualization failed: {e}")
        print("Multi-trace visualization requires graphviz. Please install graphviz.")


def _create_composite_node_label(
    multi_node: MultiTraceDAGNode, multi_dag: MultiTraceDAG
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

    data_row = f'<TR>{"".join(data_cells)}</TR>'

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


def _add_trace_legend(dot: "graphviz.Digraph", multi_dag: MultiTraceDAG) -> None:
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


def main() -> None:
    """
    Main function for the profile analysis script.
    """
    import argparse

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
Specify as <input_file1> [input_file2 ...] <output_png> <dtype>. At least 3 arguments required (1 input, output, dtype)",
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
        import shutil

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
                "Error: --visualize requires at least 3 arguments: <input_file1> <output_png> <dtype>"
            )
            return

        input_files = args.visualize[:-2]  # All but last 2 arguments
        output_png = args.visualize[-2]  # Second to last argument
        dtype = args.visualize[-1]  # Last argument

        print(
            f"Creating multi-trace DAG visualization from {len(input_files)} traces..."
        )
        print(f"Using parallel processing: {use_parallel}")
        print(f"Using caching: {use_cache}")
        if args.max_workers:
            print(f"Max workers: {args.max_workers}")

        if len(input_files) == 1:
            # Single trace visualization (backward compatibility)
            profile = JsonProfile(input_files[0], dtype=dtype)
            dag = profile.create_trace_dag_visualization(output_png)
            print(f"DAG visualization completed and saved to {output_png}")
            print(
                f"Found {len(dag.nodes)} nodes and {len(dag.edges)} edges in the trace DAG"
            )
        else:
            # Multi-trace visualization with parallel and caching options
            multi_dag = create_multi_trace_visualization(
                input_files,
                output_png,
                dtype,
                use_cache=use_cache,
                use_parallel=use_parallel,
                max_workers=args.max_workers,
            )
            print(f"Multi-trace DAG visualization completed and saved to {output_png}")
            print(
                f"Combined {len(input_files)} traces with {len(multi_dag.nodes)} unique nodes"
            )


if __name__ == "__main__":
    main()
