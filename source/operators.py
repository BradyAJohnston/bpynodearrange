# SPDX-License-Identifier: GPL-2.0-or-later

from collections.abc import Iterable
from operator import attrgetter
from statistics import fmean
from typing import Type, cast

import bpy
from bl_operators.node_editor.node_functions import node_editor_poll
from bpy.types import Context, Operator
from mathutils import Vector

from . import config
from .arrange.sugiyama import sugiyama_layout
from .utils import abs_loc, get_ntree, move


class NodeOperator:
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls: Type[Operator], context: Context) -> bool:  # type: ignore
        if not node_editor_poll(cls, context):
            return False

        if get_ntree().bl_idname == 'NodeTreeUndefined':
            cls.poll_message_set("Active node tree is undefined.")
            return False

        return True


class NA_OT_ArrangeSelected(NodeOperator, Operator):
    bl_idname = "node.na_arrange_selected"
    bl_label = "Arrange Selected"
    bl_description = "Arrange selected nodes"

    def execute(self, context: Context) -> set[str]:
        ntree = get_ntree()
        selected = [n for n in ntree.nodes if n.select]

        if not selected:
            self.report({'WARNING'}, "No nodes selected")
            return {'CANCELLED'}

        config.selected = selected
        config.SETTINGS = context.scene.na_settings  # type: ignore
        config.MARGIN = Vector(config.SETTINGS.margin).freeze()

        sugiyama_layout(ntree)

        selected.clear()
        config.linked_sockets.clear()
        config.multi_input_sort_ids.clear()
        config.SETTINGS = None

        return {'FINISHED'}


def get_all_ntrees() -> list[bpy.types.ID]:
    bl_data = []
    for key, prop in bpy.types.BlendData.bl_rna.properties.items():
        if prop.type != 'COLLECTION' or key == 'scenes':
            continue

        props = prop.fixed_type.bl_rna.properties  # type: ignore
        if {'node_tree', 'nodes'}.intersection(props.keys()):
            bl_data.extend(getattr(bpy.data, key))

    return bl_data


def batch_modify(bl_data: Iterable[bpy.types.ID], cls: Type[Operator], *, redraw_ui: bool) -> int:
    assert bpy.context

    space = cast(bpy.types.SpaceNodeEditor, bpy.context.space_data)
    path = space.path
    op = attrgetter(cls.bl_idname)(bpy.ops)
    count = 0
    for id_data in bl_data:
        if not getattr(id_data, 'use_nodes', True):
            continue

        ntree = cast(bpy.types.NodeTree, getattr(id_data, 'node_tree', id_data))
        path.append(ntree)

        if not cls.poll(bpy.context):
            path.pop()
            continue

        nodes = ntree.nodes
        old_selection = {n for n in nodes if n.select}
        for node in nodes:
            node.select = True

        if redraw_ui:
            bpy.ops.wm.redraw_timer(type='DRAW_WIN', iterations=1)

        op()
        count += 1

        for node in nodes:
            node.select = node in old_selection

        path.pop()

    return count


class NA_OT_BatchArrange(NodeOperator, Operator):
    bl_idname = "node.na_batch_arrange"
    bl_label = "Arrange Node Trees"
    bl_description = "Arrange all node trees in the current .blend file with the above settings.\nWarning: May be slow if there are expensive node trees, due to re-evaluation"

    def execute(self, context: Context):
        bl_data = get_all_ntrees()
        count = batch_modify(bl_data, NA_OT_ArrangeSelected, redraw_ui=True)
        self.report({'INFO'}, f"Arranged {count} node tree(s)")
        return {'FINISHED'}


class NA_OT_RecenterSelected(NodeOperator, Operator):
    bl_idname = "node.na_recenter_selected"
    bl_label = "Recenter Selected"
    bl_description = "Clear the locations of selected nodes"

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

        if context.scene.na_settings.origin == 'ACTIVE_NODE':  # type: ignore
            if nodes.active in non_frames:
                origin = abs_loc(nodes.active)
            else:
                self.report({'WARNING'}, "No valid active node")
                return {'CANCELLED'}
        else:
            origin = Vector(map(fmean, zip(*map(abs_loc, non_frames))))

        # Optimization: use `bpy.ops.transform.translate()` instead of modifying location
        # properties to prevent node tree re-evaluation.
        ui_scale = context.preferences.system.ui_scale
        bpy.ops.transform.translate(value=(*-origin * ui_scale, 0))

        return {'FINISHED'}


class NA_OT_BatchRecenter(NodeOperator, Operator):
    bl_idname = "node.na_batch_recenter"
    bl_label = "Recenter Node Trees"
    bl_description = "Recenter all node trees in the current .blend file with the above settings"

    def execute(self, context: Context) -> set[str]:
        bl_data = get_all_ntrees()
        count = batch_modify(bl_data, NA_OT_RecenterSelected, redraw_ui=False)
        self.report({'INFO'}, f"Recentered {count} node tree(s)")
        return {'FINISHED'}


classes = (
  NA_OT_ArrangeSelected,
  NA_OT_BatchArrange,
  NA_OT_RecenterSelected,
  NA_OT_BatchRecenter,
)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
