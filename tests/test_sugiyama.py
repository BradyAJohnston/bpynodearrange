# SPDX-License-Identifier: GPL-2.0-or-later

import pytest
import bpy
from mathutils import Vector

from bpynodearrange.arrange.sugiyama import sugiyama_layout, precompute_links
from bpynodearrange import config


class TestSugiyamaLayout:
    """Test suite for the sugiyama_layout function using real Blender node trees."""

    @pytest.fixture(autouse=True)
    def setup_blender_context(self):
        """Setup Blender context before each test."""
        # Clear existing data
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Create a new material and node tree
        mat = bpy.data.materials.new(name="TestMaterial")
        mat.use_nodes = True
        self.ntree = mat.node_tree

        # Clear default nodes
        self.ntree.nodes.clear()

        # Reset config
        config.selected = []
        config.linked_sockets.clear()
        config.multi_input_sort_ids.clear()

        yield

        # Cleanup
        bpy.data.materials.remove(mat)

    def test_sugiyama_layout_empty_tree(self):
        """Test sugiyama_layout with an empty node tree."""
        # Should handle empty tree without errors
        sugiyama_layout(self.ntree)
        assert len(self.ntree.nodes) == 0

    def test_sugiyama_layout_single_node(self):
        """Test sugiyama_layout with a single node."""
        # Create a single node
        node = self.ntree.nodes.new(type="ShaderNodeBsdfPrincipled")
        node.location = (0, 0)
        node.select = True
        config.selected = [node]

        original_location = Vector(node.location)

        sugiyama_layout(self.ntree)

        # Single node should remain in roughly the same position
        assert len(self.ntree.nodes) == 1
        # Location might change slightly due to centering, but should be reasonable
        new_location = Vector(node.location)
        assert (new_location - original_location).length < 100

    def test_sugiyama_layout_overlapping_nodes(self):
        """Test that sugiyama_layout separates overlapping nodes."""
        # Create three overlapping nodes
        node1 = self.ntree.nodes.new(type="ShaderNodeBsdfPrincipled")
        node2 = self.ntree.nodes.new(type="ShaderNodeTexImage")
        node3 = self.ntree.nodes.new(type="ShaderNodeOutputMaterial")

        # Place all nodes at the same location (overlapping)
        for node in [node1, node2, node3]:
            node.location = (0, 0)
            node.select = True

        config.selected = [node1, node2, node3]

        sugiyama_layout(self.ntree)

        # Nodes should now be separated
        locations = [Vector(node.location) for node in [node1, node2, node3]]

        # Check that nodes are no longer overlapping
        for i, loc1 in enumerate(locations):
            for j, loc2 in enumerate(locations):
                if i != j:
                    distance = (loc1 - loc2).length
                    assert distance > 10, (
                        f"Nodes {i} and {j} are still too close: {distance}"
                    )

    def test_sugiyama_layout_connected_nodes(self):
        """Test sugiyama_layout with connected nodes."""
        # Create a simple shader node chain
        tex_node = self.ntree.nodes.new(type="ShaderNodeTexImage")
        bsdf_node = self.ntree.nodes.new(type="ShaderNodeBsdfPrincipled")
        output_node = self.ntree.nodes.new(type="ShaderNodeOutputMaterial")

        # Place all at same location initially
        for node in [tex_node, bsdf_node, output_node]:
            node.location = (0, 0)
            node.select = True

        # Create connections
        self.ntree.links.new(tex_node.outputs["Color"], bsdf_node.inputs["Base Color"])
        self.ntree.links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

        config.selected = [tex_node, bsdf_node, output_node]

        sugiyama_layout(self.ntree)

        # Verify nodes are arranged in a logical flow (left to right)
        tex_x = tex_node.location[0]
        bsdf_x = bsdf_node.location[0]
        output_x = output_node.location[0]

        # Should be arranged in order: texture -> bsdf -> output
        assert tex_x < bsdf_x < output_x, (
            f"Node order incorrect: {tex_x}, {bsdf_x}, {output_x}"
        )

    def test_sugiyama_layout_with_custom_vertical_spacing(self):
        """Test sugiyama_layout with custom vertical spacing."""
        # Create multiple nodes at same location
        nodes = []
        for i in range(4):
            node = self.ntree.nodes.new(type="ShaderNodeBsdfPrincipled")
            node.location = (0, 0)
            node.select = True
            nodes.append(node)

        config.selected = nodes

        # Test with larger vertical spacing
        sugiyama_layout(self.ntree, vertical_spacing=100.0)

        # Check that nodes are spaced appropriately
        y_positions = [node.location[1] for node in nodes]
        y_positions.sort()

        # At least some vertical separation should exist
        max_y_diff = max(y_positions) - min(y_positions)
        assert max_y_diff > 50, f"Insufficient vertical spacing: {max_y_diff}"

    def test_sugiyama_layout_preserves_connections(self):
        """Test that connections between nodes are preserved after layout."""
        # Create nodes with connections
        input_node = self.ntree.nodes.new(type="ShaderNodeTexImage")
        middle_node = self.ntree.nodes.new(type="ShaderNodeBsdfPrincipled")
        output_node = self.ntree.nodes.new(type="ShaderNodeOutputMaterial")

        for node in [input_node, middle_node, output_node]:
            node.location = (0, 0)
            node.select = True

        # Create links
        self.ntree.links.new(
            input_node.outputs["Color"], middle_node.inputs["Base Color"]
        )
        self.ntree.links.new(middle_node.outputs["BSDF"], output_node.inputs["Surface"])

        config.selected = [input_node, middle_node, output_node]

        # Store original connections
        original_links = len(self.ntree.links)

        sugiyama_layout(self.ntree)

        # Verify connections are preserved
        assert len(self.ntree.links) >= original_links

        # Check that the essential connections still exist
        connected_pairs = set()
        for link in self.ntree.links:
            if link.is_valid and not link.is_hidden:
                connected_pairs.add((link.from_node, link.to_node))

        # Should maintain the logical flow
        assert any(pair[0] == input_node for pair in connected_pairs)
        assert any(pair[1] == output_node for pair in connected_pairs)

    def test_precompute_links_functionality(self):
        """Test the precompute_links function with real node tree."""
        # Create nodes with connections
        node1 = self.ntree.nodes.new(type="ShaderNodeBsdfPrincipled")
        node2 = self.ntree.nodes.new(type="ShaderNodeOutputMaterial")

        # Create a link
        link = self.ntree.links.new(node1.outputs["BSDF"], node2.inputs["Surface"])

        # Clear and precompute links
        config.linked_sockets.clear()
        precompute_links(self.ntree)

        # Verify that valid links are processed
        assert len(config.linked_sockets) > 0

        # Check that sockets are properly linked in the config
        from_socket = link.from_socket
        to_socket = link.to_socket

        if from_socket in config.linked_sockets:
            assert to_socket in config.linked_sockets[from_socket]
        if to_socket in config.linked_sockets:
            assert from_socket in config.linked_sockets[to_socket]

    def test_sugiyama_layout_with_node_frames(self):
        """Test sugiyama_layout behavior with node frames."""
        # Create regular nodes
        node1 = self.ntree.nodes.new(type="ShaderNodeBsdfPrincipled")
        node2 = self.ntree.nodes.new(type="ShaderNodeOutputMaterial")

        # Create a frame node
        frame = self.ntree.nodes.new(type="NodeFrame")
        frame.location = (0, 0)

        # Set nodes to same location
        node1.location = (0, 0)
        node2.location = (0, 0)

        # Select all nodes including frame
        for node in [node1, node2, frame]:
            node.select = True

        config.selected = [node1, node2, frame]

        # Layout should work and ignore frame nodes appropriately
        sugiyama_layout(self.ntree)

        # Regular nodes should be positioned
        assert node1.location != (0, 0) or node2.location != (0, 0)


class TestSugiyamaEdgeCases:
    """Test edge cases and error conditions for sugiyama_layout."""

    @pytest.fixture(autouse=True)
    def setup_blender_context(self):
        """Setup Blender context before each test."""
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mat = bpy.data.materials.new(name="TestMaterial")
        mat.use_nodes = True
        self.ntree = mat.node_tree
        self.ntree.nodes.clear()

        config.selected = []
        config.linked_sockets.clear()
        config.multi_input_sort_ids.clear()

        yield

        bpy.data.materials.remove(mat)

    def test_sugiyama_layout_no_selected_nodes(self):
        """Test sugiyama_layout when no nodes are selected."""
        # Create nodes but don't select them
        node1 = self.ntree.nodes.new(type="ShaderNodeBsdfPrincipled")
        node2 = self.ntree.nodes.new(type="ShaderNodeOutputMaterial")

        node1.location = (0, 0)
        node2.location = (0, 0)

        # Don't add to config.selected
        config.selected = []

        # Should handle gracefully
        sugiyama_layout(self.ntree)

        # Nodes should remain at original positions
        assert node1.location == Vector((0, 0))
        assert node2.location == Vector((0, 0))

    def test_sugiyama_layout_complex_node_network(self):
        """Test sugiyama_layout with a more complex node network."""
        # Create a more complex shader network
        tex1 = self.ntree.nodes.new(type="ShaderNodeTexImage")
        tex2 = self.ntree.nodes.new(type="ShaderNodeTexImage")
        mix = self.ntree.nodes.new(type="ShaderNodeMix")
        bsdf = self.ntree.nodes.new(type="ShaderNodeBsdfPrincipled")
        output = self.ntree.nodes.new(type="ShaderNodeOutputMaterial")

        # Place all at same location
        nodes = [tex1, tex2, mix, bsdf, output]
        for node in nodes:
            node.location = (0, 0)
            node.select = True

        # Create connections
        self.ntree.links.new(tex1.outputs["Color"], mix.inputs["A"])
        self.ntree.links.new(tex2.outputs["Color"], mix.inputs["B"])
        self.ntree.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
        self.ntree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        config.selected = nodes

        # Should arrange complex network without errors
        sugiyama_layout(self.ntree)

        # Verify nodes are no longer overlapping
        positions = [Vector(node.location) for node in nodes]

        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i != j:
                    distance = (pos1 - pos2).length
                    assert distance > 5, (
                        f"Complex network nodes {i} and {j} too close: {distance}"
                    )
