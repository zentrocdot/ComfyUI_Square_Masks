#!/usr/bin/python
'''Square mask nodes.'''

# Import the Python modules.
from PIL import Image
import numpy as np
import cv2
import torch

# Tensor to PIL function.
def tensor2pil(image):
    '''Tensor to PIL image.'''
    # Return PIL image.
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor function.
def pil2tensor(image):
    '''PIL image to tensor.'''
    # Return tensor.
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class CircleMasks:
    '''Create a circle mask in a square image.'''

    @classmethod
    def INPUT_TYPES(cls):
        '''Define the input types.'''
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "radius": ("INT", {"default": 1, "min": 1, "max": 8192}),
                "xPos": ("INT", {"default": 256, "min": 0, "max": 8192}),
                "yPos": ("INT", {"default": 256, "min": 0, "max": 8192}),
            }
        }

    RETURN_TYPES = ("MASK",)
    #RETURN_NAMES = ("MASK",)
    FUNCTION = "circle_mask_creation"
    CATEGORY = "🧬 Square Mask Nodes"
    OUTPUT_NODE = True

    def create_mask(self, width, height, radius, xpos, ypos):
        '''Create circle mask.'''
        # Create an blank image.
        maskImage = np.zeros((height, width, channels), np.uint8)
        # Draw the circumference of the circle filled.
        cv2.circle(maskImage, (width, height), radius, (255,255,255), -1)
        # Return maskImage.
        return maskImage

    def circle_mask_creation(self, width, height, radius, xpos, ypos):
        '''Main script function.'''
        # Create mask.
        image = self.create_mask()
        # Create output image.
        image_output = Image.fromarray(imgage)
        # Create tensor.
        maskImage = pil2tensor(image_output)
        # Return None.
        channel = "red"
        channels = ["red", "green", "blue", "alpha"]
        mask = maskImage[:, :, :, channels.index(channel)]
        # Return the return types.
        return (mask,)
