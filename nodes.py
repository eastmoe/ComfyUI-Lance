"""Compatibility entry point for ComfyUI-Lance nodes.

The implementation is split under ``lance_nodes`` by responsibility so this
file can remain the stable import target expected by ComfyUI.
"""

try:
    from .lance_nodes.comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    if __package__:
        raise
    # Allows direct ``import nodes`` during local smoke tests.
    from lance_nodes.comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
