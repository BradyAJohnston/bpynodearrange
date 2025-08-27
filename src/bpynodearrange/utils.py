# SPDX-License-Identifier: GPL-2.0-or-later

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from functools import cache
from operator import itemgetter
from typing import TypeVar

import bpy
from bpy.types import Node
from mathutils import Vector

from . import config


def get_ntree() -> bpy.types.NodeTree:
    return bpy.context.space_data.edit_tree  # type: ignore


_T1 = TypeVar("_T1", bound=Hashable)
_T2 = TypeVar("_T2", bound=Hashable)


def group_by(
    iterable: Iterable[_T1],
    key: Callable[[_T1], _T2],
    sort: bool = False,
) -> dict[tuple[_T1, ...], _T2]:
    groups = defaultdict(list)
    for item in iterable:
        groups[key(item)].append(item)

    items = sorted(groups.items(), key=itemgetter(0)) if sort else groups.items()
    return {tuple(group): key for key, group in items}


def abs_loc(node: Node) -> Vector:
    loc = node.location.copy()

    parent = node
    while parent := parent.parent:
        loc += parent.location

    return loc


REROUTE_DIM = Vector((8, 8))


def dimensions(node: Node) -> Vector:
    if node.bl_idname != "NodeReroute":
        return node.dimensions
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


@cache
def frame_padding() -> float:
    # Use fixed padding for headless usage
    return 27.0  # 1.5 * 18 (default widget unit)


_MAX_LOC = 100_000


def move(node: Node, *, x_offset: float = 0, y_offset: float = 0) -> None:
    if x_offset == 0 and y_offset == 0:
        return

    # If the (absolute) value of a node's X/Y axis exceeds 100k,
    # `node.location` can't be affected directly. (This often happens with
    # frames since their locations are relative.)

    loc = node.location
    if abs(loc.x + x_offset) <= _MAX_LOC and abs(loc.y + y_offset) <= _MAX_LOC:
        loc += Vector((x_offset, y_offset))
        return

    for selected_node in config.selected:
        selected_node.select = selected_node == node

    assert bpy.context
    bpy.ops.transform.translate(value=[value for value in (x_offset, y_offset, 0)])

    for selected_node in config.selected:
        selected_node.select = True
