# SPDX-License-Identifier: GPL-2.0-or-later

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from operator import itemgetter
from typing import TypeVar

import bpy
from bpy.types import Node
from mathutils import Vector

from . import config

_T1 = TypeVar('_T1', bound=Hashable)
_T2 = TypeVar('_T2', bound=Hashable)


def group_by(
  iterable: Iterable[_T1],
  key: Callable[[_T1], _T2],
  sort: bool = False,
) -> dict[tuple[_T1, ...], _T2]:
    groups = defaultdict(list)
    for item in iterable:
        groups[key(item)].append(item)

    items = sorted(groups.items(), key=itemgetter(0)) if sort else groups.items()
    return {tuple(g): k for k, g in items}


def abs_loc(node: Node) -> Vector:
    loc = node.location.copy()
    while node := node.parent:
        loc += node.location

    return loc


REROUTE_DIM = Vector((8, 8))


def dimensions(node: Node) -> Vector:
    if node.bl_idname != 'NodeReroute':
        return node.dimensions / bpy.context.preferences.system.ui_scale
    else:
        return REROUTE_DIM


_HIDE_OFFSET = 10


def get_top(node: Node, y_loc: float | None = None) -> float:
    if y_loc is None:
        y_loc = abs_loc(node).y

    return (y_loc + dimensions(node).y / 2) - _HIDE_OFFSET if node.hide else y_loc


def get_bottom(node: Node, y_loc: float | None = None) -> float:
    if y_loc is None:
        y_loc = abs_loc(node).y

    dim_y = dimensions(node).y
    bottom = y_loc - dim_y
    return bottom + dim_y / 2 - _HIDE_OFFSET if node.hide else bottom


_MAX_LOC = 100_000


def move(node: Node, *, x: float = 0, y: float = 0) -> None:
    if x == 0 and y == 0:
        return

    # If the (absolute) value of a node's X/Y axis exceeds 100k,
    # `node.location` can't be affected directly. (This often happens with
    # frames since their locations are relative.)

    loc = node.location
    if abs(loc.x + x) <= _MAX_LOC and abs(loc.y + y) <= _MAX_LOC:
        loc += Vector((x, y))
        return

    for n in config.selected:
        n.select = n == node

    ui_scale = bpy.context.preferences.system.ui_scale
    bpy.ops.transform.translate(value=[v * ui_scale for v in (x, y, 0)])

    for n in config.selected:
        n.select = True


def move_to(node: Node, *, x: float | None = None, y: float | None = None) -> None:
    loc = abs_loc(node)
    if x is not None and y is None:
        move(node, x=x - loc.x)
    elif y is not None and x is None:
        move(node, y=y - loc.y)
    else:
        move(node, x=x - loc.x, y=y - loc.y)
