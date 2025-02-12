#!/usr/bin/python
'''Square mask nodes.'''
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
#
# OpenCV co-ordinate system
#
#  0/0---column--->
#   |
#   |
#  row
#   |
#   |
#   v
#
# https://stackoverflow.com/questions/68760955/is-it-possible-to-hide-or-disable-the-turtle-screen-and-just-capture-the-final
# https://stackoverflow.com/questions/57203415/python-turtle-scrollbars

# Import the Python modules.
from tkinter import Tk, Canvas
from turtle import RawTurtle
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

    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("MASK", "INVERTED_MASK",)
    FUNCTION = "circle_mask_creation"
    CATEGORY = "🎲 Square Mask Nodes"
    OUTPUT_NODE = True

    def create_mask(self, width, height, radius, xpos, ypos):
        '''Create circle mask.'''
        # Create an blank image.
        channels = 3
        maskImage = np.zeros((height, width, channels), np.uint8)
        # Draw the circumference of the circle filled.
        cv2.circle(maskImage, (xpos, ypos), radius, (255,255,255), -1)
        # Return maskImage.
        return maskImage

    def circle_mask_creation(self, width, height, radius, xPos, yPos):
        '''Main script function.'''
        # Create mask.
        image = self.create_mask(width, height, radius, xPos, yPos)
        # Create output image.
        image_output = Image.fromarray(image)
        # Create tensor.
        maskImage = pil2tensor(image_output)
        # Return None.
        channel = "red"
        channels = ["red", "green", "blue", "alpha"]
        mask = maskImage[:, :, :, channels.index(channel)]
        invertedmask = 1 - mask
        # Return the return types.
        return (mask, invertedmask)

# ---------------------
# Function crop_image()
# ---------------------
# padding=0.025)
def crop_image(image, padding=0.0):
    '''Crop image.'''
    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get the rows and the cols from image.
    rows, cols = gray.shape
    # Get all non empty rows and cols.
    non_empty_cols = np.where(gray.min(axis=0)<255)[0]
    non_empty_rows = np.where(gray.min(axis=1)<255)[0]
    # Get the bounding box.
    dx0 = 0.0
    dx1 = 0.00001
    cropBox = (int(min(non_empty_rows) * (1 - padding)),
               int(min(max(non_empty_rows) * (1 + padding), rows)),
               int(min(non_empty_cols) * (1 - padding) - dx0),
               int(min(max(non_empty_cols) * (1 + padding + dx1), cols)))
    # Crop the image.
    cropped = image[cropBox[0]:cropBox[1], cropBox[2]:cropBox[3], :]
    # Return the cropped image.
    return cropped

# ------------------------
# Function sharp_contour()
# ------------------------
def sharp_contour(image):
    '''Sharpen contour.'''
    # Convert image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert threshold to binary image.
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # Apply morphology using a kernel.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Copy image.
    newimg = morph.copy()
    # Find all contours.
    cntrs = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    # Loop over all contours.
    for c in cntrs:
        # Draw the contour.
        cv2.drawContours(newimg,[c],0,(0,0,0),-1)
        # Calculate the approximation.
        perimeter = cv2.arcLength(c, True)
        approximation = cv2.approxPolyDP(c, 0.01 * perimeter, True)
        # Draw the contour.
        cv2.drawContours(newimg, [approximation], -1, (255, 255, 255), 3)
        newimg_out = cv2.fillPoly(newimg, pts=[approximation], color=(255,255,255))
    # Return image.
    return newimg_out

class NgonMasks:
    '''Create a circle mask in a square image.'''

    def __init__(self):
        self.wx = 512
        self.hy = 512

    @classmethod
    def INPUT_TYPES(cls):
        '''Define the input types.'''
        return {
            "required": {
                "sides": ("INT", {"default": 3, "min": 3, "max": 1024, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1024.0, "step": 0.01}),
                "alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 0.01}),
                "xPos": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "yPos": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "sharpen_contour": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "image": ("IMAGE",),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "red": ("INT", {"forceInput": True}),
                "green": ("INT", {"forceInput": True}),
                "blue": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("MASK", "INVERTED_MASK", "blank_image", "width", "height",)
    FUNCTION = "ngon_mask_creation"
    CATEGORY = "🎲 Square Mask Nodes"
    OUTPUT_NODE = True

    # Function draw_ngon()
    def draw_ngon(self, sides, alpha):
        '''Draw n-gon.'''
        # Define a function.
        def draw(angle, sides, movlen):
            # Draw polyline.
            for _ in range(sides):
                # Rotate turtle.
                ts.right(angle)
                # Move turtle.
                ts.forward(movlen)
        # Calculate the angle.
        angle = 360/sides
        # Set move length.
        movlen = 512/sides
        # Initialise turtle graphics.
        (root := Tk()).withdraw()
        canvas = Canvas(root)
        ts = RawTurtle(canvas)
        # Begin polygon.
        ts.begin_poly()
        # Set start position.
        ts.left(alpha)
        # Draw the n-gon.
        draw(angle, sides, movlen)
        # End the n-gon polygon.
        ts.end_poly()
        # Return polygon.
        return ts.get_poly()

    def ngon_mask(self, sharpen_contour, width, height, xpos, ypos, sides, alpha, scale):
        '''N-gon creator mask.'''
        # Create a blank image.
        blank_image = np.zeros((height,width,3), np.uint8)
        blank_image[:] = (255, 255, 255)
        # Draw n-gon.
        poly = self.draw_ngon(sides, alpha)
        # Move polygon.
        a = np.array([1, 1])
        b = 512
        mov = np.multiply(a,b)
        pts = np.array(poly, np.int32) + mov
        # Draw opencv image.
        color = (0, 0, 0)
        opencv_image = cv2.fillPoly(blank_image, pts=[pts], color=color)
        # Crop OpenCV image.
        cropped = crop_image(opencv_image)
        # Create PIL image.
        image_pil = Image.fromarray(cropped)
        # Resize mask image.
        newsize = (width, height)
        image_pil = image_pil.resize(newsize, resample=3)
        # Scale image.
        newsize = (int(width*scale), int(height*scale))
        image_pil = image_pil.resize(newsize, resample=1)
        # try to sharpen image.
        if sharpen_contour:
            img_np = np.array(image_pil)
            sc_img = sharp_contour(img_np)
            image_pil = Image.fromarray(sc_img)
        # Move image.
        background = Image.new('RGB', (width, height), color="white")
        offset = (xpos, ypos)
        background.paste(image_pil, offset)
        # Set mask image.
        maskImage = background
        # Return maskImage.
        return maskImage

    def ngon_mask_creation(self, sharpen_contour, xPos, yPos, sides, alpha, scale, width=512,
                           height=542, red=128, green=128, blue=128, image=None):
        '''N-gon mask creation.'''
        if image is not None:
            # Create a PIL image.
            image = tensor2pil(image)
            # Get width and height.
            width, height = image.size
        else:
            # Get width and height.
            width, height = 512, 512
        # Create a mask image.
        maskImage = self.ngon_mask(sharpen_contour, width, height, xPos, yPos, sides, alpha, scale)
        # Create a tensor.
        maskTensor = pil2tensor(maskImage)
        # Create an inverted mask.
        invertedMask = maskTensor[:, :, :, 1]
        # Create a mask.
        Mask = 1 - invertedMask
        # Create a blank image.
        blank_image = np.zeros((height,width,3), np.uint8)
        blank_image[:,0:width] = (red, green, blue)
        # Create a tensor.
        blankTensor = pil2tensor(blank_image)
        # Return the return types.
        return (Mask, invertedMask, blankTensor, width, height,)
