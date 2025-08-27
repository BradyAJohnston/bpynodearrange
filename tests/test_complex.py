import pytest
import bpy
import bpynodearrange
from pathlib import Path

PATH_TO_BLEND = Path(__file__).parent / "data" / "example.blend"


@pytest.mark.parametrize(
    "node_name", ["Style Cartoon", "Style Surface", "Simulate Elastic Network"]
)
def test_arrange_complex_tree(node_name):
    # Load the example .blend file
    bpy.ops.wm.open_mainfile(filepath=str(PATH_TO_BLEND))

    bpynodearrange.sugiyama.sugiyama_layout(bpy.data.node_groups[node_name])
