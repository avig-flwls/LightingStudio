import bpy
import argparse
import sys
from typing import Any
from pathlib import Path

class ArgumentParserForBlender(argparse.ArgumentParser):
    """Source: https://blender.stackexchange.com/a/134596.

    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self) -> list[str]:
        """This method returns the sublist right after the '--' element.

        (if present, otherwise returns an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1 :]  # the list after '--'
        except ValueError:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self) -> Any:
        return super().parse_args(args=self._get_argv_after_doubledash())


if __name__ == "__main__":
    parser = ArgumentParserForBlender()
    parser.add_argument("--experiment_folder", type=str, required=True)
    parser.add_argument("--hdri_path", type=str, required=True)
    args = parser.parse_args()


    # Collect Blender "Output" into one folder
    bpy.data.scenes["Scene"].node_tree.nodes["File Output"].base_path = str(Path(args.experiment_folder) / "blender_renders") + "/"

    #########################################################
    # Change HDRI
    #########################################################
    world = bpy.context.scene.world
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # add a texture node and load the image and link it
    texture_node = nodes.new(type="ShaderNodeTexEnvironment")
    texture_node.image = bpy.data.images.load(args.hdri_path, check_existing=True)
    texture_node.image.pack()

    # get the one background node of the world shader
    background_node = bpy.data.worlds["World"].node_tree.nodes["Background"]

    # link the new texture node to the background
    links.new(texture_node.outputs["Color"], background_node.inputs["Color"])

    # add a mapping node and a texture coordinate node
    mapping_node = nodes.new("ShaderNodeMapping")
    tex_coords_node = nodes.new("ShaderNodeTexCoord")

    # link the texture coordinate node to mapping node
    links.new(tex_coords_node.outputs["Generated"], mapping_node.inputs["Vector"])

    # link the mapping node to the texture node
    links.new(mapping_node.outputs["Vector"], texture_node.inputs["Vector"])

    ########################################################

    # Save Blender File
    bpy.ops.wm.save_as_mainfile()

    # Delete Copy
    debug_blend1_path = Path(args.experiment_folder) / "debug.blend1"
    if debug_blend1_path.exists():
        debug_blend1_path.unlink()

