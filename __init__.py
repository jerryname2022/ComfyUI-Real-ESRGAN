import os
import sys
import folder_paths

ROOT_PATH = os.path.join(folder_paths.get_folder_paths("custom_nodes")[0], "ComfyUI-Real-ESRGAN")
MODULE_GFPGAN_PATH = os.path.join(ROOT_PATH, "gfpgan")
MODULE_REALESRGAN_PATH = os.path.join(MODULE_GFPGAN_PATH, "realesrgan")

sys.path.append(ROOT_PATH)
sys.path.append(MODULE_REALESRGAN_PATH)
sys.path.append(MODULE_GFPGAN_PATH)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
