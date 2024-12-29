# SPDX-License-Identifier: GPL-2.0-or-later

from collections import defaultdict

from bpy.types import Node, NodeSocket
from mathutils import Vector

from .properties import NA_PG_Settings

selected: list[Node] = []
linked_sockets: defaultdict[NodeSocket, set[NodeSocket]] = defaultdict(set)
SETTINGS: NA_PG_Settings
MARGIN: Vector
