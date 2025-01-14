# SPDX-License-Identifier: GPL-2.0-or-later

# type: ignore

from bpy.types import Context, Panel
from bpy.utils import register_class, unregister_class

from .utils import get_ntree


class NodePanel:
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Arrange"

    @classmethod
    def poll(cls, context: Context) -> bool:
        return context.space_data.type == 'NODE_EDITOR' and get_ntree() is not None


class NA_PT_ArrangeSelected(NodePanel, Panel):
    bl_label = "Arrange"

    def draw(self, context: Context) -> None:
        layout = self.layout
        layout.use_property_split = True

        settings = context.scene.na_settings

        layout.operator("node.na_arrange_selected")
        layout.prop(settings, "margin")
        layout.prop(settings, 'balance')

        header, panel = layout.panel("batch_arrange", default_closed=True)
        header.label(text="Batch Arrange")
        if panel:
            panel.operator("node.na_batch_arrange")


class NA_PT_ClearLocations(NodePanel, Panel):
    bl_label = "Recenter"

    def draw(self, context: Context) -> None:
        layout = self.layout
        layout.use_property_split = True

        settings = context.scene.na_settings

        layout.operator("node.na_recenter_selected")
        layout.prop(settings, "origin")

        header, panel = layout.panel("batch_recenter", default_closed=True)
        header.label(text="Batch Recenter")
        if panel:
            panel.operator("node.na_batch_recenter")


classes = (
  NA_PT_ArrangeSelected,
  NA_PT_ClearLocations,
)


def register() -> None:
    for cls in classes:
        register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        unregister_class(cls)
