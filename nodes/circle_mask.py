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
        # Return the return types.
        return (mask,)


# Import the Python modules.
import turtle
from io import BytesIO
import cv2
import numpy as np
from PIL import Image, ImageFilter

# ---------------------
# Function crop_image()
# ---------------------
#def crop_image(image, padding=0.025):
def crop_image(image, padding=0.0):
    '''Crop image.'''
    # Convert image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get rows and cols from image.
    rows, cols = gray.shape
    # Get non empty rows and cols.
    non_empty_cols = np.where(gray.min(axis=0)<255)[0]
    non_empty_rows = np.where(gray.min(axis=1)<255)[0]
    # Get bounding box.
    cropBox = (int(min(non_empty_rows) * (1 - padding)),
               int(min(max(non_empty_rows) * (1 + padding), rows)),
               int(min(non_empty_cols) * (1 - padding)),
               int(min(max(non_empty_cols) * (1 + padding), cols)))
    # Crop image.
    cropped = image[cropBox[0]:cropBox[1], cropBox[2]:cropBox[3], :]
    # Return cropped image.
    return cropped

# ------------------------
# Function draw_heptagon()
# ------------------------
def draw_heptagon():
    '''Draw heptagon.'''
    # Define a function.
    def draw(angle, sides, mvlen):
        # Draw polyline.
        for _ in range(sides):
            # Rotate turtle.
            ts.right(angle)
            # Move turtle.
            ts.forward(mvlen)
    # Set move length.
    mvlen = 100
    # set speed.
    speed = 0
    # Set the background color.
    turtle.bgcolor('cyan')
    turtle.hideturtle()
    #turtle.Screen().bgcolor("orange")
    # Make a turtle object.
    ts = turtle.Turtle()
    # Hide the turtle.
    ts.hideturtle()
    # Get root screen.
    root = ts.getscreen()._root
    # Withdraw root screen.
    root.withdraw()
    # Set turtle speed.
    ts.speed(speed)
    # Set stroke size.
    ts.pensize(width=5)
    # Set sides.
    sides = 7
    # Calculate the angle.
    angle = 360/sides
    # Set the turtle graphics color.
    ts.color('blue', 'blue')
    # Set the start position.
    ts.left(angle/2)
    # Begin fill.
    ts.begin_fill()
    # Draw the heptagon.
    draw(angle, sides, mvlen)
    # End fill.
    ts.end_fill()
    # Get the screen canvas.
    cs = turtle.getscreen().getcanvas()
    # Create an eps.
    eps = cs.postscript(colormode='color')
    # Keep the window open.
    #turtle.done()
    # Return the eps.
    return eps

class HeptagonMasks:
    '''Create a circle mask in a square image.'''

    @classmethod
    def INPUT_TYPES(cls):
        '''Define the input types.'''
        return {
            "required": {
                "xPos": ("INT", {"default": 256, "min": 0, "max": 8192}),
                "yPos": ("INT", {"default": 256, "min": 0, "max": 8192}),
            }
        }

    RETURN_TYPES = ("MASK",)
    #RETURN_NAMES = ("MASK",)
    FUNCTION = "heptagon_mask_creation"
    CATEGORY = "🧬 Square Mask Nodes"
    OUTPUT_NODE = True

    def heptagon_mask(self, xpos, ypos):
        '''Create circle mask.'''
        # Draw heptagon.
        eps = draw_heptagon()
        # Create PIL image.
        pil_image = Image.open(BytesIO(eps.encode('utf-8'))).convert("RGB")
        # Create OpenCV image.
        opencv_image = np.array(pil_image)
        # Crop OpenCV image.
        cropped = crop_image(opencv_image)
        # Create PIL image.
        image_pil = Image.fromarray(cropped)
        # Resize image.
        newsize = (512, 512)
        image_pil = image_pil.resize(newsize, resample=1)
        #image_pil = image_pil.filter(ImageFilter.BoxBlur(1))
        #image_pil = image_pil.filter(ImageFilter.SMOOTH)
        image_pil = image_pil.filter(ImageFilter.BLUR)
        maskImage = image_pil
        # Save PIL image.
        #image_pil.save("turtle_image.jpg", format='JPEG')
        # Create an blank image.
        #channels = 3
        #maskImage = np.zeros((height, width, channels), np.uint8)
        # Draw the circumference of the circle filled.
        #cv2.circle(maskImage, (xpos, ypos), radius, (255,255,255), -1)
        # Return maskImage.
        return maskImage

    def heptagon_mask_creation(self, xPos, yPos):
        '''Main script function.'''
        # Create mask.
        image = self.heptagon_mask(xPos, yPos)
        # Create output image.
        #image_output = Image.fromarray(image)
        # Create tensor.
        #maskImage = pil2tensor(image_output)
        maskImage = pil2tensor(image)
        # Return None.
        channel = "red"
        channels = ["red", "green", "blue", "alpha"]
        mask = maskImage[:, :, :, channels.index(channel)]
        # Return the return types.
        return (mask,)
