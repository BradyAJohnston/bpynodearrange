# SPDX-License-Identifier: GPL-2.0-or-later

from statistics import fmean

import bpy
from bpy.types import Context, Operator
from mathutils import Vector

from . import config
from .arrange.sugiyama import sugiyama_layout
from .utils import abs_loc, get_ntree, move


class NA_OT_ArrangeSelected(Operator):
    bl_idname = "node.na_arrange_selected"
    bl_label = "Arrange Selected"
    bl_description = "Arrange selected nodes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: Context) -> set[str]:
        ntree = get_ntree()
        selected = [n for n in ntree.nodes if n.select]

        if not selected:
            self.report({'WARNING'}, "No nodes selected")
            return {'CANCELLED'}

        config.selected = selected
        config.SETTINGS = context.scene.na_settings
        config.MARGIN = Vector(config.SETTINGS.margin).freeze()

        sugiyama_layout(ntree)

        selected.clear()
        config.linked_sockets.clear()
        config.SETTINGS = None

        return {'FINISHED'}


class NA_OT_ClearLocations(Operator):
    bl_idname = "node.na_clear_locations"
    bl_label = "Clear Locations"
    bl_description = "Clear the locations of selected nodes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: Context) -> set[str]:
        nodes = get_ntree().nodes
        selected = [n for n in nodes if n.select]

        if not selected:
            self.report({'WARNING'}, "No nodes selected")
            return {'CANCELLED'}

        non_frames = [n for n in selected if n.bl_idname != 'NodeFrame']

        if not non_frames:
            self.report({'WARNING'}, "No valid nodes selected")
            return {'CANCELLED'}

        if nodes.active in non_frames:
            origin = -abs_loc(nodes.active)
        else:
            origin = -Vector(map(fmean, zip(*map(abs_loc, non_frames))))

        config.selected = selected
        for node in {n.parent or n for n in selected}:
            if node.bl_idname != 'NodeFrame' or not node.parent:
                move(node, x=origin.x, y=origin.y)

        selected.clear()
        return {'FINISHED'}


classes = (NA_OT_ArrangeSelected, NA_OT_ClearLocations)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        if cls.is_registered:
            bpy.utils.unregister_class(cls)
