# SPDX-License-Identifier: GPL-2.0-or-later

"""
Simplified Sugiyama Framework Implementation.

This module orchestrates the Sugiyama layout algorithm by coordinating
the individual phases, each handled by specialized modules.
"""

from __future__ import annotations

from statistics import fmean
from typing import cast

import networkx as nx
from bpy.types import NodeFrame, NodeTree
from mathutils import Vector

from .. import config
from ..utils import abs_loc
from .coordinates import (
    add_columns,
    assign_x_coords,
    realize_locations,
    resize_unshrunken_frame,
)
from .graph import Cluster, ClusterGraph, GNode, GType, Socket
from .multi_input import restore_multi_input_orders, save_multi_input_orders
from .ordering import minimize_crossings
from .placement.bk import bk_assign_y_coords
from .placement.linear_segments import linear_segments_assign_y_coords
from .ranking import compute_ranks
from .reroute import align_reroutes_with_sockets, realize_dummy_nodes, remove_reroutes
from .routing import route_edges


def precompute_links(ntree: NodeTree) -> None:
    """Precompute valid links in the node tree for efficient lookup."""
    config.linked_sockets.clear()
    for link in ntree.links:
        if not link.is_hidden and link.is_valid:
            config.linked_sockets[link.to_socket].add(link.from_socket)
            config.linked_sockets[link.from_socket].add(link.to_socket)


def build_graph(ntree: NodeTree) -> ClusterGraph:
    """Build the initial graph representation from the node tree."""
    # Build cluster hierarchy
    parents = {
        node.parent: Cluster(cast(NodeFrame | None, node.parent))
        for node in ntree.nodes
    }
    for cluster in parents.values():
        if cluster.node:
            cluster.cluster = parents[cluster.node.parent]

    # Create graph with nodes and edges
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
                target_node = next(
                    (target for target in graph if target.node == to_input.node), None
                )
                if target_node is None:
                    continue

                input_idx = to_input.node.inputs[:].index(to_input)
                graph.add_edge(
                    graph_node,
                    target_node,
                    from_socket=Socket(graph_node, output_idx, True),
                    to_socket=Socket(target_node, input_idx, False),
                )

    return ClusterGraph(graph)


def sugiyama_layout(
    ntree: NodeTree, vertical_spacing: float = 25.0, horizontal_spacing: float = 50.0
) -> None:
    """
    Apply the complete Sugiyama layout algorithm to nodes.

    Main orchestration function that coordinates all layout phases:
    1. Graph construction
    2. Preprocessing
    3. Ranking
    4. Crossing minimization
    5. Coordinate assignment
    6. Edge routing
    7. Realization
    """
    # Get layout nodes and preserve center
    layout_nodes = [node for node in ntree.nodes if node.bl_idname != "NodeFrame"]
    locations = [abs_loc(node) for node in layout_nodes]
    if not locations:
        return

    old_center = Vector(map(fmean, zip(*locations)))

    # Clear and initialize config
    config.multi_input_sort_ids.clear()

    # Phase 1: Graph Construction
    precompute_links(ntree)
    cluster_graph = build_graph(ntree)
    graph = cluster_graph.G
    tree = cluster_graph.T

    # Phase 2: Preprocessing
    save_multi_input_orders(graph, ntree)
    remove_reroutes(cluster_graph)

    # Phase 3: Ranking
    compute_ranks(cluster_graph)
    cluster_graph.merge_edges()
    cluster_graph.insert_dummy_nodes()

    # Phase 4: Crossing Minimization
    add_columns(graph)
    minimize_crossings(graph, tree)

    # Phase 5: Y-Coordinate Assignment
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

    # Phase 6: Coordinate Assignment and Routing
    align_reroutes_with_sockets(cluster_graph)
    assign_x_coords(graph, tree, horizontal_spacing)
    route_edges(graph, tree, horizontal_spacing / 2, vertical_spacing / 2)

    # Phase 7: Realization
    realize_dummy_nodes(cluster_graph, ntree)
    restore_multi_input_orders(graph, ntree)
    realize_locations(graph, old_center, ntree)

    # Finalize frame sizes
    for cluster in cluster_graph.S:
        resize_unshrunken_frame(cluster_graph, cluster)
