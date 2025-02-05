# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from bpy.types import Node, NodeSocket
from mathutils import Vector

from .properties import NA_PG_Settings

if TYPE_CHECKING:
    from .arrange.graph import Socket

selected: list[Node] = []
linked_sockets: defaultdict[NodeSocket, set[NodeSocket]] = defaultdict(set)
multi_input_sort_ids: defaultdict[Socket, dict[Socket, int]] = defaultdict(dict)

SETTINGS: NA_PG_Settings
MARGIN: Vector
