"""
DAG node classes for representing trace execution graphs.
"""

from typing import Dict, List, Tuple

from torch.utils._ordered_set import OrderedSet


try:
    import graphviz
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


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

        # Roofline analysis
        self.bound_type_list: List[str] = (
            []
        )  # List of "compute" or "memory" for each instance based on roofline analysis

        # Multi-trace support
        self.trace_data: Dict[int, Dict] = {}  # Maps trace_id to trace-specific data


class MultiTraceDAGNode:
    """Represents a composite node in the multi-trace DAG."""

    def __init__(self, name: str, node_type: str):
        self.name = name
        self.node_type = node_type  # 'op' or 'kernel'
        self.trace_instances: Dict[int, TraceDAGNode] = {}  # Maps trace_id to node data
        self.present_in_traces: OrderedSet[int] = (
            OrderedSet()
        )  # Which traces contain this node

    def add_trace_instance(self, trace_id: int, node: TraceDAGNode):
        """Add data for this node from a specific trace."""
        self.trace_instances[trace_id] = node
        self.present_in_traces.add(trace_id)


class MultiTraceDAG:
    """Directed Acyclic Graph representing multiple collapsed trace trees."""

    def __init__(self):
        self.nodes: Dict[str, MultiTraceDAGNode] = {}
        self.edges: OrderedSet[Tuple[str, str, int]] = (
            OrderedSet()
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
        min_intensity = 0.01  # Minimum 1% of original color (colorless)
        max_intensity = 1.2  # 120% of original color (slightly darker)

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
        self.edges: OrderedSet[Tuple[str, str]] = (
            OrderedSet()
        )  # (parent, child) relationships

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
