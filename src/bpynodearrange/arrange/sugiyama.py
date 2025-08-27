# SPDX-License-Identifier: GPL-2.0-or-later

"""
Sugiyama Framework Implementation for Node Layout in Blender.

This module implements the Sugiyama framework, a well-known algorithm for drawing
directed acyclic graphs in a hierarchical layout. It's specifically adapted for
arranging nodes in Blender's node editor with support for node frames, reroute
nodes, and multi-input sockets.

The Sugiyama framework consists of four main phases:
1. Cycle removal and ranking assignment
2. Edge crossings minimization
3. Coordinate assignment (x and y positioning)
4. Edge routing and bend point insertion

This implementation includes extensions for:
- Clustering support via node frames
- Reroute node optimization and alignment
- Multi-input socket order preservation
- Edge bend point generation to avoid node overlaps

References
----------
Sugiyama, K., Tagawa, S., & Toda, M. (1981). Methods for visual understanding
of hierarchical system structures. IEEE Transactions on Systems, Man, and
Cybernetics, 11(2), 109-125.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Iterator, Sequence
from itertools import chain
from statistics import fmean
from typing import cast

import networkx as nx
from bpy.types import Node, NodeFrame, NodeTree
from mathutils import Vector
from mathutils.geometry import intersect_line_line_2d

from .. import config
from ..utils import abs_loc, frame_padding, group_by
from .graph import (
    FROM_SOCKET,
    TO_SOCKET,
    Cluster,
    ClusterGraph,
    GNode,
    GType,
    MultiEdge,
    Socket,
    add_dummy_edge,
    add_dummy_nodes_to_edge,
    is_real,
    lowest_common_cluster,
    socket_graph,
)
from .ordering import minimize_crossings
from .placement.bk import bk_assign_y_coords
from .placement.linear_segments import linear_segments_assign_y_coords
from .ranking import compute_ranks

# -------------------------------------------------------------------


def precompute_links(ntree: NodeTree) -> None:
    """
    Precompute valid links in the node tree for efficient lookup.

    Builds a lookup table of connected sockets by iterating through all links
    in the node tree and storing only valid, visible connections. This avoids
    repeatedly checking link validity and reduces lookup time complexity.

    Parameters
    ----------
    ntree : NodeTree
        The Blender node tree to precompute links for.

    Notes
    -----
    Results are stored in `config.linked_sockets` as a defaultdict mapping
    each socket to a set of connected sockets. Only links that are both
    valid and not hidden are included.
    """

    for link in ntree.links:
        if not link.is_hidden and link.is_valid:
            config.linked_sockets[link.to_socket].add(link.from_socket)
            config.linked_sockets[link.from_socket].add(link.to_socket)


def get_multidigraph(ntree: NodeTree) -> nx.MultiDiGraph[GNode]:
    """
    Create a MultiDiGraph representation of all nodes in the tree and their connections.

    Converts all Blender nodes into a NetworkX MultiDiGraph where nodes become
    GNode objects and links become directed edges with socket information. Excludes
    NodeFrame nodes from the graph while preserving their clustering relationships.

    Parameters
    ----------
    ntree : NodeTree
        The Blender node tree to process

    Returns
    -------
    nx.MultiDiGraph[GNode]
        A directed multigraph where:
        - Nodes are GNode objects wrapping all Blender nodes (excluding frames)
        - Edges represent connections between node sockets
        - Edge data includes 'from_socket' and 'to_socket' Socket objects

    Notes
    -----
    This function also builds the cluster hierarchy by associating each node
    with its parent frame (if any) and establishing parent-child relationships
    between nested frames.
    """
    parents = {
        node.parent: Cluster(cast(NodeFrame | None, node.parent))
        for node in ntree.nodes
    }
    for cluster in parents.values():
        if cluster.node:
            cluster.cluster = parents[cluster.node.parent]

    graph = nx.MultiDiGraph()
    graph.add_nodes_from(
        [
            GNode(node, parents[node.parent])
            for node in ntree.nodes
            if node.bl_idname != "NodeFrame"
        ]
    )
    for graph_node in graph:
        for output_idx, from_output in enumerate(graph_node.node.outputs):
            for to_input in config.linked_sockets[from_output]:
                # Only include connections between nodes in the graph
                target_node = next(
                    (target for target in graph if target.node == to_input.node), None
                )
                if target_node is None:
                    continue

                # target_node already found above
                input_idx = to_input.node.inputs[:].index(to_input)
                graph.add_edge(
                    graph_node,
                    target_node,
                    from_socket=Socket(graph_node, output_idx, True),
                    to_socket=Socket(target_node, input_idx, False),
                )

    return graph


def get_nesting_relations(
    vertex: GNode | Cluster,
) -> Iterator[tuple[Cluster, GNode | Cluster]]:
    """
    Generate all nesting relationships for a vertex up the cluster hierarchy.

    Recursively yields parent-child relationships starting from the given vertex
    and moving up the cluster hierarchy until reaching the root level.

    Parameters
    ----------
    vertex : GNode | Cluster
        The vertex to find nesting relations for.

    Yields
    ------
    tuple[Cluster, GNode | Cluster]
        Pairs of (parent_cluster, child_vertex) representing the nesting hierarchy.

    Examples
    --------
    If node A is in frame F1, which is in frame F2:
    >>> list(get_nesting_relations(node_A))
    [(F1, node_A), (F2, F1)]
    """
    if cluster := vertex.cluster:
        yield (cluster, vertex)
        yield from get_nesting_relations(cluster)


def save_multi_input_orders(graph: nx.MultiDiGraph[GNode], ntree: NodeTree) -> None:
    """
    Save the current ordering of connections to multi-input sockets.

    Multi-input sockets in Blender can have multiple connections, and their
    order affects evaluation. This function preserves the current connection
    order so it can be restored after layout operations.

    Parameters
    ----------
    graph : nx.MultiDiGraph[GNode]
        The node graph containing the connections.
    ntree : NodeTree
        The Blender node tree.

    Notes
    -----
    For reroute nodes, traces back to find the original source socket to
    preserve the correct ordering relationship. Results are stored in
    `config.multi_input_sort_ids` for later restoration.
    """
    links = {(link.from_socket, link.to_socket): link for link in ntree.links}
    for from_node, to_node, edge_data in graph.edges.data():
        to_socket = edge_data[TO_SOCKET]

        if not to_socket.bpy.is_multi_input:
            continue

        if from_node.is_reroute:
            for current_node, prev_node in chain(
                [(to_node, from_node)], nx.bfs_edges(graph, from_node, reverse=True)
            ):
                if not prev_node.is_reroute:
                    break
            base_from_socket = graph.edges[prev_node, current_node, 0][FROM_SOCKET]
        else:
            base_from_socket = edge_data[FROM_SOCKET]

        link = links[(edge_data[FROM_SOCKET].bpy, to_socket.bpy)]
        config.multi_input_sort_ids[to_socket].append(
            (base_from_socket, link.multi_input_sort_id)
        )


# -------------------------------------------------------------------


def get_reroute_paths(
    cluster_graph: ClusterGraph,
    function: Callable | None = None,
    *,
    preserve_reroute_clusters: bool = True,
    must_be_aligned: bool = False,
) -> list[list[GNode]]:
    """
    Find connected chains of reroute nodes that can be processed together.

    Identifies sequences of reroute nodes that form linear paths, taking into
    account cluster boundaries and alignment constraints. Used for optimization
    operations like reroute removal or alignment.

    Parameters
    ----------
    cluster_graph : ClusterGraph
        The cluster graph containing the nodes.
    function : Callable, optional
        Filter function to apply to reroute nodes. Only nodes passing this
        filter will be included in paths.
    preserve_reroute_clusters : bool, default=True
        If True, prevents paths from crossing cluster boundaries where both
        clusters contain only reroute nodes.
    must_be_aligned : bool, default=False
        If True, only includes edges between reroutes at the same y-coordinate.

    Returns
    -------
    list[list[GNode]]
        List of reroute paths, where each path is a list of connected reroute
        nodes sorted in topological order. Paths are sorted by their starting
        node's position in the graph.

    Notes
    -----
    The function removes edges from reroutes that have multiple outputs,
    keeping only linear chains. This prevents branching in the paths.
    """
    graph = cluster_graph.G
    reroutes = {
        vertex
        for vertex in graph
        if vertex.is_reroute and (not function or function(vertex))
    }
    subgraph = nx.DiGraph(graph.subgraph(reroutes))

    for vertex in subgraph:
        if graph.out_degree[vertex] > 1:
            subgraph.remove_edges_from(tuple(subgraph.out_edges(vertex)))

    if preserve_reroute_clusters:
        reroute_clusters = {  #
            cluster
            for cluster in cluster_graph.S
            if all(
                vertex.is_reroute
                for vertex in cluster_graph.T[cluster]
                if vertex.type != GType.CLUSTER
            )
        }
        subgraph.remove_edges_from(
            [  #
                (from_node, to_node)
                for from_node, to_node in subgraph.edges
                if from_node.cluster != to_node.cluster
                and {from_node.cluster, to_node.cluster} & reroute_clusters
            ]
        )

    if must_be_aligned:
        subgraph.remove_edges_from(
            [
                (from_node, to_node)
                for from_node, to_node in subgraph.edges
                if from_node.y != to_node.y
            ]
        )

    indices = {
        vertex: i
        for i, vertex in enumerate(nx.topological_sort(graph))
        if vertex in reroutes
    }
    paths = [
        sorted(component, key=lambda vertex: indices[vertex])
        for component in nx.weakly_connected_components(subgraph)
    ]
    paths.sort(key=lambda path: indices[path[0]])
    return paths


def is_safe_to_remove(vertex: GNode) -> bool:
    """
    Check if a reroute node can be safely removed from the graph.

    A reroute node is safe to remove if:
    - It's not a real node (dummy node)
    - It has no label
    - It's not used in multi-input socket ordering

    Parameters
    ----------
    vertex : GNode
        The node to check for removal safety.

    Returns
    -------
    bool
        True if the node can be safely removed, False otherwise.

    Notes
    -----
    This function is typically used before dissolving reroute paths to
    ensure important connectivity information isn't lost.
    """
    if not is_real(vertex):
        return True

    if vertex.node.label:
        return False

    for sort_values in config.multi_input_sort_ids.values():
        if any(vertex == sort_item[0].owner for sort_item in sort_values):
            return False

    # Since we're processing all nodes in the tree, we can safely remove reroutes
    # that don't have labels and aren't used in multi-input ordering
    return True


def dissolve_reroute_edges(graph: nx.DiGraph[GNode], path: list[GNode]) -> None:
    """
    Remove a reroute path by connecting its inputs directly to its outputs.

    Takes a linear path of reroute nodes and replaces it with direct connections
    from the path's input to all of its outputs, effectively removing the
    intermediate reroute nodes.

    Parameters
    ----------
    graph : nx.DiGraph[GNode]
        The graph containing the reroute path.
    path : list[GNode]
        Linear sequence of reroute nodes to dissolve.

    Notes
    -----
    The function also creates corresponding Blender node links to match
    the graph structure. If the same output is already connected to the
    same input (through different paths), the path is cleared to avoid
    duplicate connections.
    """
    if not graph[path[-1]]:
        return

    try:
        predecessor, _, output_socket = next(
            iter(graph.in_edges(path[0], data=FROM_SOCKET))
        )
    except StopIteration:
        return

    successor_inputs = [edge[2] for edge in graph.out_edges(path[-1], data=TO_SOCKET)]

    # Check if a reroute has been used to link the same output to the same multi-input multiple
    # times
    for *_, edge_data in graph.out_edges(predecessor, data=True):
        if (
            edge_data[FROM_SOCKET] == output_socket
            and edge_data[TO_SOCKET] in successor_inputs
        ):
            path.clear()
            return

    for input_socket in successor_inputs:
        graph.add_edge(
            predecessor,
            input_socket.owner,
            from_socket=output_socket,
            to_socket=input_socket,
        )
        input_socket.id_data.links.new(output_socket.bpy, input_socket.bpy)


def remove_reroutes(cluster_graph: ClusterGraph) -> None:
    """
    Remove unnecessary reroute nodes from the cluster graph.

    Processes reroute paths and either removes them completely (by dissolving)
    or simplifies them by removing intermediate nodes while preserving
    connectivity. Handles special cases for reroute-only clusters.

    Parameters
    ----------
    cluster_graph : ClusterGraph
        The cluster graph to process.

    Notes
    -----
    For clusters containing only reroute nodes, long paths are simplified
    to just their endpoints with dummy edges. For mixed clusters, entire
    safe reroute paths are dissolved.
    """
    reroute_clusters = {  #
        cluster
        for cluster in cluster_graph.S
        if all(
            vertex.type != GType.CLUSTER and vertex.is_reroute
            for vertex in cluster_graph.T[cluster]
        )
    }
    for path in get_reroute_paths(cluster_graph, is_safe_to_remove):
        if path[0].cluster in reroute_clusters:
            if len(path) > 2:
                start_node, *intermediate_nodes, end_node = path
                add_dummy_edge(cluster_graph.G, start_node, end_node)
                cluster_graph.remove_nodes_from(intermediate_nodes)
        else:
            dissolve_reroute_edges(cluster_graph.G, path)
            cluster_graph.remove_nodes_from(path)


# -------------------------------------------------------------------


def add_columns(graph: nx.DiGraph[GNode]) -> None:
    """
    Organize nodes into columns based on their rank and sort by position.

    Groups nodes by their rank (horizontal layer) and sorts them within
    each column by their current y-coordinate. This establishes the basic
    columnar structure for the layout.

    Parameters
    ----------
    graph : nx.DiGraph[GNode]
        The directed graph to organize into columns.

    Notes
    -----
    Results are stored in `graph.graph['columns']` as a list of node lists.
    Each node also gets a reference to its column in the `column` attribute.
    Real nodes are sorted by their actual y-position, while dummy nodes
    default to y=0.
    """
    columns = [
        list(component)
        for component in group_by(graph, key=lambda vertex: vertex.rank, sort=True)
    ]
    graph.graph["columns"] = columns
    for column in columns:
        column.sort(
            key=lambda vertex: abs_loc(vertex.node).y if is_real(vertex) else 0,
            reverse=True,
        )
        for vertex in column:
            vertex.col = column


# -------------------------------------------------------------------


def align_reroutes_with_sockets(cluster_graph: ClusterGraph) -> None:
    """
    Align reroute nodes with their connected sockets for cleaner routing.

    Adjusts the vertical positions of reroute paths to minimize visual
    clutter by aligning them with the sockets they connect to. Uses an
    iterative process to find optimal alignments.

    Parameters
    ----------
    cluster_graph : ClusterGraph
        The cluster graph containing reroute paths to align.

    Notes
    -----
    The alignment process considers:
    - Distance to connected sockets
    - Collision avoidance with other nodes in the same column
    - Preference for exact y-coordinate matches with connected sockets

    The algorithm iteratively refines positions until no more improvements
    can be made or all alignment opportunities are exhausted.
    """
    reroute_paths: dict[tuple[GNode, ...], list[Socket]] = {}
    for path in get_reroute_paths(
        cluster_graph, preserve_reroute_clusters=False, must_be_aligned=True
    ):
        inputs = cluster_graph.G.in_edges(path[0], data=FROM_SOCKET)
        outputs = cluster_graph.G.out_edges(path[-1], data=TO_SOCKET)
        reroute_paths[tuple(path)] = [edge[2] for edge in (*inputs, *outputs)]

    while True:
        changed = False
        for path, foreign_sockets in tuple(reroute_paths.items()):
            current_y = path[0].y
            foreign_sockets.sort(key=lambda socket: abs(current_y - socket.y))
            foreign_sockets.sort(
                key=lambda socket: current_y == socket.owner.y, reverse=True
            )

            if not foreign_sockets or current_y - foreign_sockets[0].y == 0:
                del reroute_paths[path]
                continue

            movement = current_y - foreign_sockets[0].y
            current_y -= movement
            if movement < 0:
                above_y_vals = [
                    (above_node := vertex.col[vertex.col.index(vertex) - 1]).y
                    - above_node.height
                    for vertex in path
                    if vertex != vertex.col[0]
                ]
                if above_y_vals and current_y > min(above_y_vals):
                    continue
            else:
                below_y_vals = [
                    vertex.col[vertex.col.index(vertex) + 1].y
                    for vertex in path
                    if vertex != vertex.col[-1]
                ]
                if below_y_vals and max(below_y_vals) > current_y - path[0].height:
                    continue

            for vertex in path:
                vertex.y -= movement

            changed = True

        if not changed:
            if reroute_paths:
                for path, foreign_sockets in reroute_paths.items():
                    del foreign_sockets[0]
            else:
                break


def frame_padding_of_col(
    columns: Sequence[Collection[GNode]],
    column_index: int,
    tree: nx.DiGraph[GNode | Cluster],
) -> float:
    """
    Calculate additional spacing needed between columns due to frame nesting.

    Determines extra horizontal spacing required between adjacent columns
    when they contain nodes in different frame hierarchies. The spacing
    accounts for the visual nesting depth of frames.

    Parameters
    ----------
    columns : Sequence[Collection[GNode]]
        List of all columns in the layout.
    column_index : int
        Index of the current column.
    tree : nx.DiGraph[GNode | Cluster]
        The cluster tree representing frame hierarchies.

    Returns
    -------
    float
        Additional spacing in pixels needed between column column_index and column_index+1.

    Notes
    -----
    Uses the longest path algorithm on cluster differences to determine
    the maximum nesting depth that needs to be visually represented.
    """
    current_column = columns[column_index]

    if current_column == columns[-1]:
        return 0

    clusters1 = {cast(Cluster, vertex.cluster) for vertex in current_column}
    clusters2 = {cast(Cluster, vertex.cluster) for vertex in columns[column_index + 1]}

    if not clusters1 ^ clusters2:
        return 0

    subtree1 = tree.subgraph(
        chain(clusters1, *[nx.ancestors(tree, cluster) for cluster in clusters1])
    ).copy()
    subtree2 = tree.subgraph(
        chain(clusters2, *[nx.ancestors(tree, cluster) for cluster in clusters2])
    ).copy()

    for *edge_nodes, edge_data in subtree1.edges(data=True):
        edge_data["weight"] = int(edge_nodes not in subtree2.edges)  # type: ignore

    for *edge_nodes, edge_data in subtree2.edges(data=True):
        edge_data["weight"] = int(edge_nodes not in subtree1.edges)  # type: ignore

    distance = nx.dag_longest_path_length(subtree1) + nx.dag_longest_path_length(
        subtree2
    )  # type: ignore
    return frame_padding() * distance


def assign_x_coords(
    graph: nx.DiGraph[GNode], tree: nx.DiGraph[GNode | Cluster], x_spacing: float = 50.0
) -> None:
    """
    Assign horizontal coordinates to all nodes based on their columns.

    Positions nodes horizontally by column, with dynamic spacing that
    accounts for node widths, edge bend requirements, and frame nesting.
    Includes intelligent spacing adjustments for long edges.

    Parameters
    ----------
    graph : nx.DiGraph[GNode]
        The graph with nodes organized into columns.
    tree : nx.DiGraph[GNode | Cluster]
        The cluster tree for frame hierarchy information.
    x_spacing : float, default=50.0
        Base horizontal spacing between columns in pixels.

    Notes
    -----
    Uses adaptive spacing based on edge characteristics - longer edges
    (indicating more complex routing) get additional spacing. Reroute
    nodes are positioned at the left edge of their column.
    """
    columns: list[list[GNode]] = graph.graph["columns"]
    current_x = 0
    for column_index, column in enumerate(columns):
        max_width = max([vertex.width for vertex in column])

        for vertex in column:
            vertex.x = (
                current_x
                if vertex.is_reroute
                else current_x - (vertex.width - max_width) / 2
            )

        # https://doi.org/10.7155/jgaa.00220 (p. 139)
        delta_spacing = sum(
            [
                1
                for *_, edge_data in graph.out_edges(column, data=True)
                if abs(edge_data[TO_SOCKET].y - edge_data[FROM_SOCKET].y)
                >= x_spacing * 3
            ]
        )
        spacing = (1 + min(delta_spacing / 4, 2)) * x_spacing
        current_x += (
            max_width + spacing + frame_padding_of_col(columns, column_index, tree)
        )


def is_unnecessary_bend_point(
    socket: Socket,
    other_socket: Socket,
    x_spacing: float = 25.0,
    y_spacing: float = 25.0,
) -> bool:
    """
    Determine if a bend point would be unnecessary for edge routing.

    Checks whether adding a bend point at a socket would actually help
    avoid node overlaps or if the edge can be routed cleanly without it.

    Parameters
    ----------
    socket : Socket
        The socket where a potential bend point would be placed.
    other_socket : Socket
        The socket at the other end of the edge.
    x_spacing : float, default=25.0
        Horizontal spacing buffer around nodes.
    y_spacing : float, default=25.0
        Vertical spacing buffer around nodes.

    Returns
    -------
    bool
        True if the bend point would be unnecessary, False if it's needed.

    Notes
    -----
    The function considers neighboring nodes in the same column and checks
    if a direct line between sockets would intersect with any node boundaries
    including frame padding.
    """
    vertex = socket.owner

    if vertex.is_reroute:
        return False

    vertex_index = vertex.col.index(vertex)
    is_above = other_socket.y > socket.y

    try:
        neighbor = (
            vertex.col[vertex_index - 1] if is_above else vertex.col[vertex_index + 1]
        )
    except IndexError:
        return True

    if neighbor.is_reroute:
        return True

    neighbor_x_offset, neighbor_y_offset = x_spacing / 2, y_spacing / 2
    neighbor_y = (
        neighbor.y - neighbor.height - neighbor_y_offset
        if is_above
        else neighbor.y + neighbor_y_offset
    )

    assert neighbor.cluster
    if neighbor.cluster.node and neighbor.cluster != vertex.cluster:
        neighbor_x_offset += frame_padding()
        if is_above:
            neighbor_y -= frame_padding()
        else:
            neighbor_y += frame_padding() + neighbor.cluster.label_height()

    line_a = (
        (neighbor.x - neighbor_x_offset, neighbor_y),
        (neighbor.x + neighbor.width + neighbor_x_offset, neighbor_y),
    )
    line_b = ((socket.x, socket.y), (other_socket.x, other_socket.y))
    return intersect_line_line_2d(*line_a, *line_b) is None


_MIN_X_DIFF = 30
_MIN_Y_DIFF = 15


def add_bend_points(
    graph: nx.MultiDiGraph[GNode],
    vertex: GNode,
    bend_points: defaultdict[MultiEdge, list[GNode]],
    x_spacing: float = 25.0,
    y_spacing: float = 25.0,
) -> None:
    """
    Add bend points for edges connected to a node to avoid visual conflicts.

    Creates dummy nodes positioned at the edge of a node to provide clean
    edge routing. Bend points help edges avoid overlapping with node bodies
    and provide cleaner visual connections.

    Parameters
    ----------
    graph : nx.MultiDiGraph[GNode]
        The graph containing the edges.
    vertex : GNode
        The node to process for bend point creation.
    bend_points : defaultdict[MultiEdge, list[GNode]]
        Dictionary to store bend points for each edge.
    x_spacing : float, default=25.0
        Horizontal spacing for collision detection.
    y_spacing : float, default=25.0
        Vertical spacing for collision detection.

    Notes
    -----
    Bend points are only created when:
    - The horizontal distance is significant enough to warrant routing
    - The vertical distance between sockets is substantial
    - The bend point would actually help avoid visual conflicts
    """
    edge_data: dict[str, Socket]
    largest = max(vertex.col, key=lambda node: node.width)
    for from_node, to_node, key, edge_data in (
        *graph.out_edges(vertex, data=True, keys=True),
        *graph.in_edges(vertex, data=True, keys=True),
    ):
        socket = edge_data[FROM_SOCKET] if vertex == from_node else edge_data[TO_SOCKET]
        bend_point = GNode(type=GType.DUMMY)
        bend_point.x = largest.x + largest.width if socket.is_output else largest.x

        if abs(socket.x - bend_point.x) <= _MIN_X_DIFF:
            continue

        bend_point.y = socket.y
        other_socket = next(sock for sock in edge_data.values() if sock != socket)

        if abs(other_socket.y - bend_point.y) <= _MIN_Y_DIFF:
            continue

        if is_unnecessary_bend_point(socket, other_socket, x_spacing, y_spacing):
            continue

        bend_points[from_node, to_node, key].append(bend_point)


def node_overlaps_edge(
    vertex: GNode,
    edge_line: tuple[tuple[float, float], tuple[float, float]],
) -> bool:
    """
    Check if a node's bounding box intersects with an edge line.

    Tests whether a straight line edge would visually overlap with a node's
    rectangular boundary, used to determine if edge routing is needed.

    Parameters
    ----------
    vertex : GNode
        The node to check for overlap.
    edge_line : tuple[tuple[float, float], tuple[float, float]]
        Line segment defined by two (x, y) coordinate pairs.

    Returns
    -------
    bool
        True if the edge line intersects the node's bounding box.

    Notes
    -----
    Only checks top and bottom edges of the node rectangle. Reroute nodes
    are considered non-overlapping since they're small routing elements.
    """
    if vertex.is_reroute:
        return False

    top_line = ((vertex.x, vertex.y), (vertex.x + vertex.width, vertex.y))
    if intersect_line_line_2d(*edge_line, *top_line):
        return True

    bottom_line = (
        (vertex.x, vertex.y - vertex.height),
        (vertex.x + vertex.width, vertex.y - vertex.height),
    )
    if intersect_line_line_2d(*edge_line, *bottom_line):
        return True

    return False


def route_edges(
    graph: nx.MultiDiGraph[GNode],
    tree: nx.DiGraph[GNode | Cluster],
    x_spacing: float = 25.0,
    y_spacing: float = 25.0,
) -> None:
    """
    Create bend points for all edges to enable clean visual routing.

    Processes the entire graph to add bend points where needed, optimizes
    bend point sharing between similar edges, and handles special cases
    like reroute fan-out patterns.

    Parameters
    ----------
    graph : nx.MultiDiGraph[GNode]
        The graph containing edges to route.
    tree : nx.DiGraph[GNode | Cluster]
        The cluster tree for hierarchy information.
    x_spacing : float, default=25.0
        Horizontal spacing for bend point placement.
    y_spacing : float, default=25.0
        Vertical spacing for bend point placement.

    Notes
    -----
    The routing process includes several optimizations:
    - Merges bend points that serve the same routing purpose
    - Handles reroute nodes with multiple outputs specially
    - Reuses bend points for edges that can share routing paths
    - Adds dummy nodes to the graph structure for visualization
    """
    bend_points = defaultdict(list)
    for vertex in chain(*graph.graph["columns"]):
        add_bend_points(graph, vertex, bend_points, x_spacing, y_spacing)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    edge_of = {
        bend_point: edge
        for edge, dummy_list in bend_points.items()
        for bend_point in dummy_list
    }

    def bend_point_key(bend_point: GNode) -> tuple[Socket, float, float]:
        return (
            graph.edges[edge_of[bend_point]][FROM_SOCKET],
            bend_point.x,
            bend_point.y,
        )

    for (target, *redundant), (from_socket, *_) in group_by(
        edge_of, key=bend_point_key
    ).items():
        for bend_point in redundant:
            dummy_nodes = bend_points[edge_of[bend_point]]
            dummy_nodes[dummy_nodes.index(bend_point)] = target

        owner_node = from_socket.owner
        if not owner_node.is_reroute or graph.out_degree[owner_node] < 2:  # type: ignore
            continue

        for edge in graph.out_edges(owner_node, keys=True):
            if (
                target not in bend_points[edge]
                and graph.edges[edge][TO_SOCKET].y == target.y
            ):
                bend_points[edge].append(target)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    for edge, dummy_nodes in tuple(bend_points.items()):
        dummy_nodes.sort(key=lambda bend_point: bend_point.x)
        from_socket = graph.edges[edge][FROM_SOCKET]
        for other_edge in graph.out_edges(edge[0], keys=True):
            edge_data = graph.edges[other_edge]

            if edge_data[FROM_SOCKET] != from_socket or other_edge in bend_points:
                continue

            if edge_data[TO_SOCKET].x <= dummy_nodes[-1].x:
                continue

            last_bend_point = dummy_nodes[-1]
            line = (
                (last_bend_point.x, last_bend_point.y),
                (edge_data[TO_SOCKET].x, edge_data[TO_SOCKET].y),
            )
            if any(node_overlaps_edge(vertex, line) for vertex in edge[1].col):
                continue

            bend_points[other_edge] = dummy_nodes

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    lowest_common_clusters = lowest_common_cluster(tree, bend_points)
    for (from_node, to_node, key), dummy_nodes in bend_points.items():
        add_dummy_nodes_to_edge(graph, (from_node, to_node, key), dummy_nodes)

        cluster = lowest_common_clusters.get((from_node, to_node), from_node.cluster)
        for dummy_node in dummy_nodes:
            dummy_node.cluster = cluster
            tree.add_edge(cluster, dummy_node)


# -------------------------------------------------------------------


def simplify_path(cluster_graph: ClusterGraph, path: list[GNode]) -> None:
    """
    Simplify a reroute path by removing unnecessary intermediate nodes.

    Optimizes reroute paths by connecting endpoints directly when possible,
    eliminating redundant intermediate reroute nodes while preserving
    connectivity.

    Parameters
    ----------
    cluster_graph : ClusterGraph
        The cluster graph containing the path.
    path : list[GNode]
        The reroute path to simplify.

    Notes
    -----
    The simplification considers three cases:
    1. Input-aligned: Connect predecessor directly to end node
    2. Output-aligned: Connect start node directly to successor
    3. General case: Add dummy edge between endpoints

    Removed nodes are also eliminated from the original path list.
    """
    if len(path) == 1:
        return

    start_node, *intermediate_nodes, end_node = path
    graph = cluster_graph.G

    if (
        graph.pred[start_node]
        and (
            input_socket := next(iter(graph.in_edges(start_node, data=FROM_SOCKET)))[2]
        ).y
        == start_node.y
    ):
        graph.add_edge(
            input_socket.owner,
            end_node,
            from_socket=input_socket,
            to_socket=Socket(end_node, 0, False),
        )
        intermediate_nodes.append(start_node)
    elif (
        graph.out_degree[end_node] == 1
        and end_node.y
        == (output_socket := next(iter(graph.out_edges(end_node, data=TO_SOCKET)))[2]).y
    ):
        graph.add_edge(
            start_node,
            output_socket.owner,
            from_socket=Socket(start_node, 0, True),
            to_socket=output_socket,
        )
        intermediate_nodes.append(end_node)
    elif intermediate_nodes:
        add_dummy_edge(graph, start_node, end_node)

    cluster_graph.remove_nodes_from(intermediate_nodes)
    for node in intermediate_nodes:
        if node not in graph:
            path.remove(node)


def add_reroute(vertex: GNode) -> None:
    """
    Convert a dummy node into a real Blender reroute node.

    Creates an actual NodeReroute in Blender and associates it with the
    dummy graph node, effectively "realizing" the dummy node.

    Parameters
    ----------
    vertex : GNode
        The dummy node to convert to a real reroute.

    Notes
    -----
    The created reroute inherits the cluster assignment (parent frame)
    from the dummy node and is added to the selected nodes list.
    """
    reroute = vertex.node.id_data.nodes.new(type="NodeReroute")
    assert vertex.cluster
    reroute.parent = vertex.cluster.node
    vertex.node = reroute
    vertex.type = GType.NODE


def realize_edges(graph: nx.DiGraph[GNode], vertex: GNode) -> None:
    """
    Create actual Blender node links for edges connected to a realized node.

    Takes graph edges and creates corresponding Blender NodeLinks to match
    the graph structure in the actual node tree.

    Parameters
    ----------
    graph : nx.DiGraph[GNode]
        The graph containing edge information.
    vertex : GNode
        The realized node to create links for.

    Notes
    -----
    Only creates links between real nodes (not dummy nodes). For reroute
    nodes, uses their single input/output sockets.
    """
    assert is_real(vertex)
    links = vertex.node.id_data.links

    if graph.pred[vertex]:
        predecessor_output = next(iter(graph.in_edges(vertex, data=FROM_SOCKET)))[2]
        links.new(predecessor_output.bpy, vertex.node.inputs[0])

    for _, successor_node, successor_input in graph.out_edges(vertex, data=TO_SOCKET):
        if is_real(successor_node):
            links.new(vertex.node.outputs[0], successor_input.bpy)


def realize_dummy_nodes(cluster_graph: ClusterGraph) -> None:
    """
    Convert all dummy nodes in reroute paths to actual Blender reroute nodes.

    Processes aligned reroute paths, simplifies them where possible, and
    converts any remaining dummy nodes to real reroute nodes with proper
    Blender node links.

    Parameters
    ----------
    cluster_graph : ClusterGraph
        The cluster graph containing dummy nodes to realize.

    Notes
    -----
    This is a critical step in the layout process that bridges between
    the abstract graph representation and the concrete Blender node tree.
    """
    for path in get_reroute_paths(
        cluster_graph, is_safe_to_remove, must_be_aligned=True
    ):
        simplify_path(cluster_graph, path)

        for vertex in path:
            if not is_real(vertex):
                add_reroute(vertex)

            realize_edges(cluster_graph.G, vertex)


def restore_multi_input_orders(graph: nx.MultiDiGraph[GNode], ntree: NodeTree) -> None:
    """
    Restore the original connection order for multi-input sockets.

    Uses previously saved multi-input socket orders to recreate the original
    connection sequence after layout operations may have changed the link order.

    Parameters
    ----------
    graph : nx.MultiDiGraph[GNode]
        The graph containing the current connections.
    ntree : NodeTree, optional
        The Blender node tree

    Notes
    -----
    This function handles complex cases involving reroute nodes by tracing
    through the socket graph to find the correct connection mappings.
    If sort IDs aren't unique, it recreates all links to ensure proper ordering.
    """
    links = ntree.links
    socket_g = socket_graph(graph)
    for socket, sort_ids in config.multi_input_sort_ids.items():
        multi_input = socket.bpy
        assert multi_input

        socket_links = {
            link.from_socket: link for link in links if link.to_socket == multi_input
        }

        if len(socket_links) != len(
            {link.multi_input_sort_id for link in socket_links.values()}
        ):
            for link in socket_links.values():
                links.remove(link)

            for output in socket_links:
                socket_links[output] = links.new(output, multi_input)

        socket_subgraph = socket_g.subgraph(
            {sort_item[0] for sort_item in sort_ids}
            | {socket}
            | {vertex for vertex in socket_g if vertex.owner.is_reroute}
        )
        seen_sockets = set()
        for base_from_socket, sort_id in sort_ids:
            matching_link = min(
                socket_links.values(),
                key=lambda link: abs(link.multi_input_sort_id - sort_id),
            )
            from_socket = next(
                source
                for source, target in nx.edge_dfs(socket_subgraph, base_from_socket)
                if target == socket and source not in seen_sockets
            )
            socket_links[from_socket.bpy].swap_multi_input_sort_id(matching_link)  # type: ignore
            seen_sockets.add(from_socket)


def realize_locations(
    graph: nx.DiGraph[GNode], old_center: Vector, ntree: NodeTree
) -> None:
    """
    Apply computed node positions to actual Blender nodes.

    Transfers the calculated x,y coordinates from the graph nodes to their
    corresponding Blender nodes, maintaining the overall center position.

    Parameters
    ----------
    graph : nx.DiGraph[GNode]
        The graph with computed node positions.
    old_center : Vector
        The original center point of the selected nodes.

    Notes
    -----
    The function:
    - Temporarily removes parent assignments for optimization
    - Calculates offset to preserve the original center
    - Restores parent assignments after positioning
    """
    new_center = (
        fmean([vertex.x for vertex in graph]),
        fmean([vertex.y for vertex in graph]),
    )
    offset_x, offset_y = -Vector(new_center) + old_center

    for vertex in graph:
        assert isinstance(vertex.node, Node)
        assert vertex.cluster

        # Optimization: avoid using bpy.ops for as many nodes as possible (see `utils.move()`)
        vertex.node.parent = None

        current_x, current_y = vertex.node.location
        vertex.x += offset_x
        vertex.y += offset_y
        vertex.node.location = (vertex.x, vertex.corrected_y())

        vertex.node.parent = vertex.cluster.node


def resize_unshrunken_frame(cluster_graph: ClusterGraph, cluster: Cluster) -> None:
    """
    Resize node frames that are set to not shrink automatically.

    For frames with shrink=False, temporarily enables shrinking to allow
    the frame to resize to fit its contents, then restores the setting.

    Parameters
    ----------
    cluster_graph : ClusterGraph
        The cluster graph containing the frame.
    cluster : Cluster
        The cluster representing the frame to resize.

    Notes
    -----
    This workaround is needed because Blender frames with shrink=False
    don't automatically adjust their size when child nodes are moved.
    """
    frame = cluster.node

    if not frame or frame.shrink:
        return

    real_children = [vertex for vertex in cluster_graph.T[cluster] if is_real(vertex)]

    for vertex in real_children:
        vertex.node.parent = None

    frame.shrink = False
    frame.shrink = True

    for vertex in real_children:
        vertex.node.parent = frame


# -------------------------------------------------------------------


def sugiyama_layout(
    ntree: NodeTree, vertical_spacing: float = 50.0, horizontal_spacing: float = 50.0
) -> None:
    """
    Apply the complete Sugiyama layout algorithm to selected nodes.

    This is the main entry point that orchestrates the full Sugiyama framework
    layout process for Blender nodes. It handles the complete pipeline from
    graph construction through final node positioning and link creation.

    Parameters
    ----------
    ntree : NodeTree
        The Blender node tree to layout.
    vertical_spacing : float, default=50.0
        Vertical spacing between node rows in pixels.
    horizontal_spacing : float, default=50.0
        Horizontal spacing between node columns in pixels.

    Notes
    -----
    The algorithm follows these main phases:

    1. **Graph Construction**: Convert selected nodes to graph representation
    2. **Preprocessing**: Handle reroutes and save multi-input orders
    3. **Ranking**: Assign hierarchical ranks to nodes
    4. **Crossing Minimization**: Reduce edge crossings between layers
    5. **Coordinate Assignment**: Position nodes with proper spacing
    6. **Edge Routing**: Add bend points for clean edge visualization
    7. **Realization**: Convert graph back to actual Blender nodes and links

    The function preserves the original center point of the selected nodes
    and handles special cases like frame hierarchies and multi-input sockets.

    If no nodes are selected or no valid layout can be computed, the function
    returns early without making changes.

    Examples
    --------
    Basic usage with default spacing:

    >>> sugiyama_layout(bpy.context.space_data.node_tree)

    Custom spacing for tighter layout:

    >>> sugiyama_layout(ntree, vertical_spacing=30, horizontal_spacing=40)
    """
    # Get all non-frame nodes for layout
    layout_nodes = [node for node in ntree.nodes if node.bl_idname != "NodeFrame"]
    locations = [abs_loc(node) for node in layout_nodes]

    if not locations:
        return

    old_center = Vector(map(fmean, zip(*locations)))

    # Clear config to ensure clean state
    config.linked_sockets.clear()
    config.multi_input_sort_ids.clear()

    precompute_links(ntree)
    cluster_graph = ClusterGraph(get_multidigraph(ntree))
    graph = cluster_graph.G
    tree = cluster_graph.T

    save_multi_input_orders(graph, ntree)
    remove_reroutes(cluster_graph)

    compute_ranks(cluster_graph)
    cluster_graph.merge_edges()
    cluster_graph.insert_dummy_nodes()

    add_columns(graph)
    minimize_crossings(graph, tree)

    if len(cluster_graph.S) == 1:
        bk_assign_y_coords(graph, vertical_spacing=vertical_spacing)
    else:
        cluster_graph.add_vertical_border_nodes()
        linear_segments_assign_y_coords(
            cluster_graph, vertical_spacing=vertical_spacing
        )
        cluster_graph.remove_nodes_from(
            [vertex for vertex in graph if vertex.type == GType.VERTICAL_BORDER]
        )

    align_reroutes_with_sockets(cluster_graph)
    assign_x_coords(graph, tree, horizontal_spacing)
    route_edges(graph, tree, horizontal_spacing / 2, vertical_spacing / 2)

    realize_dummy_nodes(cluster_graph)
    restore_multi_input_orders(graph, ntree)
    realize_locations(graph, old_center, ntree)
    for cluster in cluster_graph.S:
        resize_unshrunken_frame(cluster_graph, cluster)
