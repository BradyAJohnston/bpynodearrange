# SPDX-License-Identifier: GPL-2.0-or-later

# https://link.springer.com/chapter/10.1007/3-540-36151-0_26
# https://doi.org/10.1016/j.jvlc.2013.11.005
# https://doi.org/10.7155/jgaa.00088

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from itertools import chain
from math import inf
from random import seed, uniform

import networkx as nx

from .graph import Cluster, GNode, GNodeType, Socket

# -------------------------------------------------------------------


def get_col_nesting_trees(columns: Sequence[Collection[GNode]], T: nx.DiGraph) -> list[nx.DiGraph]:
    trees = []
    for col in columns:
        LT = nx.DiGraph()
        nodes = set(chain(col, *[nx.ancestors(T, v) for v in col]))
        LT.add_edges_from([(u, v) for u in nodes for v in T[u] if v in nodes])
        trees.append(LT)

    return trees


@cache
def reflexive_transitive_closure(LT: nx.DiGraph) -> nx.DiGraph:
    return nx.transitive_closure(LT, reflexive=True)


@cache
def topologically_sorted_clusters(LT: nx.DiGraph) -> list[Cluster]:
    return [h for h in nx.topological_sort(LT) if h.type == GNodeType.CLUSTER]


@dataclass(slots=True)
class ClusterCrossingsData:
    free_col: list[GNode | Cluster]
    fixed_sockets: dict[GNode, tuple[Socket, ...]] = field(default_factory=dict)
    free_sockets: dict[GNode | Cluster, tuple[Socket, ...]] = field(default_factory=dict)
    N: list[Socket] = field(default_factory=list)
    S: list[Socket] = field(default_factory=list)
    bipartite_edges: list[tuple[Socket, Socket, int]] = field(default_factory=list)


def crossing_reduction_data(
  G: nx.DiGraph,
  trees: Sequence[nx.DiGraph],
  backwards: bool = False,
) -> Iterator[list[ClusterCrossingsData]]:
    for LT in trees[1:]:
        TC = reflexive_transitive_closure(LT)
        data = []
        for h in topologically_sorted_clusters(LT):
            G_h = nx.DiGraph()
            G_h.add_nodes_from(LT[h])
            for s, t, d in G.in_edges(TC[h], data=True):
                c = next(c for c in TC.pred[t] if c in LT[h])

                if (s, c) in G_h.edges:
                    G_h.edges[s, c]['weight'] += 1
                    continue

                k1 = 'from_socket'
                k2 = 'to_socket'
                if d[k1].owner != s:
                    k1, k2 = k2, k1

                attr = dict(from_socket=replace(d[k1], owner=s), to_socket=replace(d[k2], owner=c))
                G_h.add_edge(s, c, weight=1, **attr)

            # -------------------------------------------------------------------

            H = ClusterCrossingsData(list(LT[h]))

            for u in chain(*[G_h.pred[v] for v in LT[h]]):
                sockets = {e[2] for e in G_h.out_edges(u, data='from_socket')}
                H.fixed_sockets[u] = sorted(sockets, key=lambda d: d.idx, reverse=not backwards)

            for v in LT[h]:
                H.free_sockets[v] = [e[2] for e in G_h.in_edges(v, data='from_socket')]

            # -------------------------------------------------------------------

            B = nx.DiGraph()
            edges = [(d['from_socket'], d['to_socket'], d) for *_, d in G_h.edges.data()]
            B.add_edges_from(edges)

            if B.edges:
                N, S = map(set, zip(*B.edges))
                if len(S) > len(N):
                    N, S = S, N
                    B = nx.reverse_view(B)

                H.N.extend(sorted(N, key=lambda d: d.idx))
                H.S.extend(sorted(S, key=lambda d: d.idx))

            H.bipartite_edges.extend(B.edges.data('weight'))

            # -------------------------------------------------------------------

            data.append(H)

        yield data


# -------------------------------------------------------------------

_FreeColumns = Sequence[tuple[list[GNode], nx.DiGraph, Sequence[ClusterCrossingsData]]]
_RANDOM_AMOUNT = 0.07


def calc_socket_ranks(H: ClusterCrossingsData, forwards: bool) -> None:
    for v, sockets in H.fixed_sockets.items():
        incr = 1 / (len(sockets) + 1)
        rank = v.col.index(v) + 1
        if forwards:
            incr *= -1

        for socket in sockets:
            rank += incr
            v.cr.socket_ranks[socket] = rank


def calc_barycenters(H: ClusterCrossingsData) -> None:
    for w in H.free_col:
        sockets = H.free_sockets[w]

        if not sockets:
            continue

        weight = sum([s.owner.cr.socket_ranks[s] for s in sockets])
        weight += uniform(0, 1) * _RANDOM_AMOUNT - _RANDOM_AMOUNT / 2
        w.cr.barycenter = weight / len(sockets)


def fill_in_unknown_barycenters(col: list[GNode], is_first_iter: bool) -> None:
    if is_first_iter:
        max_b = max([b for v in col if (b := v.cr.barycenter) is not None], default=0) + 2
        for v in col:
            if v.cr.barycenter is None:
                v.cr.barycenter = uniform(0, 1) * max_b - 1
        return

    for i, v in enumerate(col):
        if v.cr.barycenter is not None:
            continue

        prev_b = col[i - 1].cr.barycenter if i != 0 else 0
        next_b = next((b for w in col[i + 1:] if (b := w.cr.barycenter) is not None), prev_b + 1)
        v.cr.barycenter = (prev_b + next_b) / 2


def get_cross_count(H: ClusterCrossingsData) -> int:
    edges = H.bipartite_edges

    if not edges:
        return 0

    free_col = set(H.free_col)

    def pos(w: Socket) -> float:
        v = w.owner
        return v.cr.barycenter if v in free_col else v.col.index(v)

    H.N.sort(key=pos)
    H.S.sort(key=pos)

    south_indicies = {k: i for i, k in enumerate(H.S)}
    north_indicies = {k: i for i, k in enumerate(H.N)}

    edges.sort(key=lambda e: south_indicies[e[1]])
    edges.sort(key=lambda e: north_indicies[e[0]])

    first_idx = 1
    while first_idx < len(H.S):
        first_idx *= 2

    tree = [0] * (2 * first_idx - 1)
    first_idx -= 1

    cross_weight = 0
    for _, v, weight in edges:
        idx = south_indicies[v] + first_idx
        tree[idx] += weight
        weight_sum = 0
        while idx > 0:
            if idx % 2 == 1:
                weight_sum += tree[idx + 1]

            idx = (idx - 1) // 2
            tree[idx] += weight

        cross_weight += weight * weight_sum

    return cross_weight


def get_new_col_order(v: GNode | Cluster, LT: nx.DiGraph) -> Iterator[GNode]:
    if v.type == GNodeType.CLUSTER:
        for w in sorted(LT[v], key=lambda w: w.cr.barycenter):
            yield from get_new_col_order(w, LT)
    else:
        yield v


def sort_internal_columns(items: _FreeColumns) -> None:
    for base_free_col, LT, data in items:

        def key(v: GNode | Cluster) -> int:
            if v.type == GNodeType.CLUSTER:
                descendants = nx.descendants(LT, v)
                w = next(w for w in base_free_col if w in descendants)
            else:
                w = v

            return base_free_col.index(w)

        for H in data:
            H.free_col.sort(key=key)


# -------------------------------------------------------------------

_ITERATIONS = 15


def minimized_crossing_count(
  columns: Sequence[list[GNode]],
  forward_items: _FreeColumns,
  backward_items: _FreeColumns,
) -> int:
    nodes_and_clusters = tuple(chain(*[i[1] for i in forward_items]))
    i = -1
    cross_count = inf
    while True:
        for v in nodes_and_clusters:
            v.cr.reset()

        i += 1
        old_cross_count = cross_count

        if cross_count == 0:
            break

        cross_count = 0
        items = forward_items if i % 2 == 0 else backward_items
        for base_free_col, LT, data in items:
            for H in data:
                calc_socket_ranks(H, i % 2 == 0)
                calc_barycenters(H)
                fill_in_unknown_barycenters(H.free_col, i == 0)
                cross_count += get_cross_count(H)

            root = topologically_sorted_clusters(LT)[0]
            new_order = tuple(get_new_col_order(root, LT))
            base_free_col.sort(key=new_order.index)

        if old_cross_count > cross_count:
            sort_internal_columns(forward_items + backward_items)
            best_columns = [c.copy() for c in columns]
        else:
            for col, best_col in zip(columns, best_columns):
                col.sort(key=best_col.index)
            break

    return old_cross_count


def minimize_crossings(G: nx.DiGraph, T: nx.DiGraph) -> None:
    columns = G.graph['columns']
    trees = get_col_nesting_trees(columns, T)

    forward_data = crossing_reduction_data(G, trees)
    forward_items = list(zip(columns[1:], trees[1:], forward_data))

    trees.reverse()
    backward_data = crossing_reduction_data(nx.reverse_view(G), trees, True)
    backward_items = list(zip(columns[-2::-1], trees[1:], backward_data))

    # -------------------------------------------------------------------

    seed(0)
    best_cross_count = inf
    best_columns = [c.copy() for c in columns]
    for _ in range(_ITERATIONS):
        cross_count = minimized_crossing_count(columns, forward_items, backward_items)
        if cross_count < best_cross_count:
            best_cross_count = cross_count
            best_columns = [c.copy() for c in columns]
            if best_cross_count == 0:
                break
        else:
            for col, best_col in zip(columns, best_columns):
                col.sort(key=best_col.index)
