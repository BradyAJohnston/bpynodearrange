# SPDX-License-Identifier: GPL-2.0-or-later

# type: ignore

from bpy.props import BoolProperty, IntVectorProperty, PointerProperty
from bpy.types import PropertyGroup, Scene
from bpy.utils import register_class, unregister_class


class NA_PG_Settings(PropertyGroup):
    margin: IntVectorProperty(
      name="Spacing",
      description="Space between nodes",
      default=(50, 50),
      min=0,
      options=set(),
      subtype='XYZ',
      size=2)

    balance: BoolProperty(
      name="Balance",
      description="Reduce link lengths by vertically positioning nodes between their neighbours",
      default=True,
      options=set())


def register() -> None:
    register_class(NA_PG_Settings)
    Scene.na_settings = PointerProperty(type=NA_PG_Settings)


def unregister() -> None:
    if NA_PG_Settings.is_registered:
        unregister_class(NA_PG_Settings)
    del Scene.na_settings
