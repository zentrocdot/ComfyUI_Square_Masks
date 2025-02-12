from .nodes.circle_mask import *

NODE_CLASS_MAPPINGS = { 
    "⚗️ Circle Masks": CircleMasks,
    "⚗️ Ngon Masks": NgonMasks,
    }
    
WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

MESSAGE = "\033[34mComfyUI Square Masks Nodes: \033[92mLoaded\033[0m" 

print(MESSAGE)
