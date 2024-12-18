# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from math import inf
from typing import TYPE_CHECKING

from bpy.types import Node, NodeFrame, NodeSocket
from mathutils import Vector
from mathutils.geometry import interpolate_bezier

from ..utils import REROUTE_DIM, abs_loc, dimensions, get_bottom, get_top

if TYPE_CHECKING:
    from .placement.linear_segments import Segment


class GNodeType(Enum):
    NODE = auto()
    DUMMY = auto()
    CLUSTER = auto()
    HORIZONTAL_BORDER = auto()
    VERTICAL_BORDER = auto()


@dataclass(slots=True)
class CrossingReduction:
    socket_ranks: dict[Socket, float] = field(default_factory=dict)
    barycenter: float | None = None

    def reset(self) -> None:
        self.socket_ranks.clear()
        self.barycenter = None


class GNode:
    node: Node | None
    cluster: Cluster | None
    type: GNodeType

    is_reroute: bool
    width: float
    height: float

    rank: int | None
    po_num: int | None
    lowest_po_num: int | None

    col: list[GNode] | None
    cr: CrossingReduction

    x: float | None
    y: float | None

    segment: Segment | GNode

    root: GNode
    aligned: GNode
    cells: tuple[list[int], list[float]]
    sink: GNode
    shift: float

    __slots__ = tuple(__annotations__)

    def __init__(
      self,
      node: Node | None = None,
      cluster: Cluster | None = None,
      type: GNodeType = GNodeType.NODE,
      rank: int | None = None,
    ) -> None:
        self.node = node
        self.cluster = cluster
        self.type = type
        self.rank = rank
        self.is_reroute = type == GNodeType.DUMMY or (
          self.is_real and node.bl_idname == 'NodeReroute')

        if self.is_reroute:
            self.width = REROUTE_DIM.x
            self.height = REROUTE_DIM.y
        elif self.is_real:
            self.width = dimensions(node).x
            self.height = get_top(node) - get_bottom(node)
        else:
            self.width = 0
            self.height = 0

        self.po_num = None
        self.lowest_po_num = None

        self.col = None
        self.cr = CrossingReduction()

        self.x = None
        self.reset()

        self.segment = self

    def __hash__(self) -> int:
        return id(self)

    @property
    def is_real(self) -> bool:
        return isinstance(self.node, Node)

    def reset(self) -> None:
        self.root = self
        self.aligned = self
        self.cells = None

        self.sink = self
        self.shift = inf
        self.y = None

    def corrected_y(self) -> float:
        return self.y + (abs_loc(self.node).y - get_top(self.node))


@dataclass(slots=True)
class Cluster:
    node: NodeFrame | None
    cluster: Cluster | None = None

    nesting_level: int | None = None
    left: GNode | None = None
    right: GNode | None = None
    cr: CrossingReduction = field(default_factory=CrossingReduction)

    def __hash__(self) -> int:
        return id(self)

    @property
    def type(self) -> GNodeType:
        return GNodeType.CLUSTER


_HIDDEN_NODE_FLAT_WIDTH = 116
_BOTTOM_OFFSET = 14.85
_TOP_OFFSET = 35
_VISIBLE_PBSDF_SOCKETS = 5
_SOCKET_SPACING_MULTIPLIER = 22


@dataclass(frozen=True)
class Socket:
    owner: GNode | Cluster
    idx: int
    is_output: bool

    @property
    def bpy(self) -> NodeSocket:
        v = self.owner

        if not v.is_real:
            return None

        sockets = v.node.outputs if self.is_output else v.node.inputs
        return sockets[self.idx]

    def x(self) -> float:
        v = self.owner
        return v.x + v.width if self.is_output else v.x

    def _get_hidden_socket_y(self) -> float:
        v = self.owner
        socket = self.bpy
        node = socket.node

        cap_width = (v.width - _HIDDEN_NODE_FLAT_WIDTH) / 2
        outer = cap_width / -3

        raw_sockets = node.outputs if self.is_output else node.inputs
        sockets = [s for s in raw_sockets if not s.is_unavailable]

        bottom = v.y - v.height
        coords = ((cap_width, v.y), (outer, v.y), (cap_width, bottom), (outer, bottom))
        points = interpolate_bezier(*map(Vector, coords), len(sockets) + 2)[1:-1]

        return points[sockets.index(socket)].y

    def _get_input_y(self) -> float:
        input = self.bpy
        node = input.node

        if node.hide:
            return self._get_hidden_socket_y()

        v = self.owner
        y = v.y

        inputs = [i for i in node.inputs if not i.is_unavailable and not i.hide]
        if node.bl_idname != 'ShaderNodeBsdfPrincipled':
            # Start from the bottom socket to avoid any node properties
            y -= v.height - _BOTTOM_OFFSET
            inputs.reverse()
            idx = inputs.index(input)

            for i in inputs[:idx + 1]:
                is_multi_value = i.type in {'VECTOR', 'ROTATION', 'MATRIX'}
                if is_multi_value and not i.hide_value and not i.is_linked:
                    y += (_SOCKET_SPACING_MULTIPLIER * 0.909) * len(i.default_value)
        else:
            y -= 56.5
            idx = -inputs.index(input)
            if idx < -_VISIBLE_PBSDF_SOCKETS:
                panels = ('Subsurface', 'Specular', 'Transmission', 'Coat', 'Sheen', 'Emission')
                idx = -(_VISIBLE_PBSDF_SOCKETS + 0.5 + panels.index(input.name.split()[0]))

        return y + idx * _SOCKET_SPACING_MULTIPLIER

    def _get_output_y(self) -> float:
        output = self.bpy
        node = output.node

        if node.hide:
            return self._get_hidden_socket_y()

        y = self.owner.y - _TOP_OFFSET
        outputs = [o for o in node.outputs if not o.is_unavailable and not o.hide]
        return y - outputs.index(output) * _SOCKET_SPACING_MULTIPLIER

    @cached_property
    def _offset_y(self) -> float:
        if self.owner.is_reroute or not self.owner.is_real:
            return 0

        socket_y = self._get_output_y() if self.is_output else self._get_input_y()
        return socket_y - self.owner.y

    @property
    def y(self) -> float:
        return self.owner.y + self._offset_y
