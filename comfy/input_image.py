import numpy as np
import torch
import cv2
import lunar_tools as lt
from ..tools.input_image import InputImageProcessor, AcidProcessor
from PIL import Image

class LRInputImageProcessor:
    """
    LRInputImageProcessor is a class that provides various image processing functionalities such as 
    adjusting brightness, saturation, hue rotation, blurring, and human segmentation. It leverages 
    the InputImageProcessor class to perform these operations.

    Methods:
        __init__(): Initializes the LRInputImageProcessor instance and sets up the InputImageProcessor.
        
        INPUT_TYPES(): Class method that returns a dictionary specifying the required input types 
                       for the image processing operations. The inputs include image, brightness, 
                       saturation, hue rotation angle, blur kernel size, and flags for enabling 
                       blur, human segmentation, infrared imaging and image flipping.
                       
    Attributes:
        iip (InputImageProcessor): An instance of the InputImageProcessor class used to perform 
                                   the actual image processing operations.
    """
    def __init__(self):
        self.iip = InputImageProcessor()
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "brightness": ("FLOAT", {
                    "default": 1, 
                    "min": 0,
                    "max": 3,
                    "step": 1e-2,
                    "display": "number"
                }),
                "saturization": ("FLOAT", {
                    "default": 1, 
                    "min": 0,
                    "max": 2,
                    "step": 1e-2,
                    "display": "number"
                }),
                "hue_rotation_angle": ("FLOAT", {
                    "default": 0, 
                    "min": 0,
                    "max": 180,
                    "step": 1,
                    "display": "number"
                }),
                "blur_kernel_size": ("INT", {
                    "default": 3, 
                    "min": 1,
                    "max": 25,
                    "step": 2,
                    "display": "number"
                }),
                "do_blur": ("BOOLEAN", {"default": False}),
                "is_infrared": ("BOOLEAN", {"default": False}),
                "do_human_seg": ("BOOLEAN", {"default": False}),
                "flip_axis": ("INT", {
                    "default": -1, 
                    "min": -1,
                    "max": 2,
                    "step": 1,
                    "display": "number"
                }),                
                "resizing_factor_humanseg": ("FLOAT", {
                    "default": 0.4, 
                    "min": 0,
                    "max": 1,
                    "step": 1e-2,
                    "display": "number"
                })
            },
        }
            
    RETURN_TYPES = ("IMAGE", "IMAGE",)  
    RETURN_NAMES = ("processed image", "human segmentation mask", )
    FUNCTION = "process"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/visual"

    def process(
        self, 
        image, 
        brightness=None, 
        saturization=None, 
        hue_rotation_angle=None, 
        blur_kernel_size=None, 
        do_blur=None, 
        is_infrared=None, 
        do_human_seg=None,
        flip_axis=None,
        resizing_factor_humanseg=None
    ):
        if brightness is not None:
            self.iip.set_brightness(brightness)
        if saturization is not None:
            self.iip.set_saturization(saturization)
        if hue_rotation_angle is not None:
            self.iip.set_hue_rotation(hue_rotation_angle)
        if blur_kernel_size is not None:
            self.iip.set_blur_size(blur_kernel_size)
        if do_blur is not None:
            self.iip.set_blur(do_blur)
        if is_infrared is not None:
            self.iip.set_infrared(is_infrared)
        if do_human_seg is not None:
            self.iip.set_human_seg(do_human_seg)
        if flip_axis:
            flip_axis = np.clip(flip_axis, -1, 2)
            flip_axis = int(flip_axis)
            if flip_axis == -1:
                self.iip.set_flip(False, 0)
            else:
                self.iip.set_flip(True, flip_axis)
        else:
            self.iip.set_flip(False)
        if resizing_factor_humanseg is not None:
            self.iip.set_resizing_factor_humanseg(resizing_factor_humanseg)
        
        image, human_segmmask = self.iip.process(image)
        
        image = [image, human_segmmask]
        return image

    

class LRAcidProcessor:
    def __init__(self):
        self.ap = AcidProcessor()
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "current_input_image": ("IMAGE", {}),
                "last_diffusion_image": ("IMAGE", {}),

                "acid_strength": ("FLOAT", {
                    "default": 0.05, 
                    "min": 0,
                    "max": 1,
                    "step": 1e-2,
                    "display": "number"
                }),
                "acid_strength_foreground": ("FLOAT", {
                    "default": 0.01, 
                    "min": 0,
                    "max": 1,
                    "step": 1e-2,
                    "display": "number"
                }),
                "coef_noise": ("FLOAT", {
                    "default": 0.15, 
                    "min": 0,
                    "max": 3,
                    "step": 1e-2,
                    "display": "number"
                }),
                "x_shift": ("INT", {
                    "default": 0, 
                    "min": -30,
                    "max": 30,
                    "step": 1,
                    "display": "number"
                }),
                "y_shift": ("INT", {
                    "default": 0, 
                    "min": -30,
                    "max": 30,
                    "step": 1,
                    "display": "number"
                }),
                "zoom_factor": ("FLOAT", {
                    "default": 1, 
                    "min": 0.5,
                    "max": 2,
                    "step": 1e-2,
                    "display": "number"
                }),
                "rotation_angle": ("FLOAT", {
                    "default": 0, 
                    "min": -180,
                    "max": 180,
                    "step": 1,
                    "display": "number"
                }),
                "do_acid_tracers": ("BOOLEAN", {"default": False}),
                "do_apply_humansegm_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "human_segmentation_mask": ("IMAGE", {}),
            },
        }
            
    RETURN_TYPES = ("IMAGE", )  
    RETURN_NAMES = ("acid image", )
    FUNCTION = "process"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/visual"

    def process(
        self, 
        current_input_image,
        last_diffusion_image=None, 
        acid_strength=None,
        acid_strength_foreground=None,
        coef_noise=None,
        x_shift=None,
        y_shift=None,
        zoom_factor=None,
        rotation_angle=None,
        do_acid_tracers=None,
        do_apply_humansegm_mask=None,
        human_segmentation_mask=None,
    ):
        if last_diffusion_image is not None:
            self.ap.update(last_diffusion_image)
        if acid_strength is not None:
            self.ap.set_acid_strength(acid_strength)
        if acid_strength_foreground is not None:
            self.ap.set_acid_strength_foreground(acid_strength_foreground)
        if coef_noise is not None:
            self.ap.set_coef_noise(coef_noise)
        if x_shift is not None:
            self.ap.set_x_shift(x_shift)
        if y_shift is not None:
            self.ap.set_y_shift(y_shift)
        if zoom_factor is not None and zoom_factor != 0:
            self.ap.set_zoom_factor(zoom_factor)
        if rotation_angle is not None:
            self.ap.set_rotation_angle(rotation_angle)
        if do_acid_tracers is not None:
            self.ap.set_do_acid_tracers(do_acid_tracers)
        if do_apply_humansegm_mask is not None:
            self.ap.set_apply_humansegm_mask(do_apply_humansegm_mask)
        if human_segmentation_mask is not None:
            self.ap.set_human_segmmask(human_segmentation_mask)
        image = self.ap.process(current_input_image)
        image = [image]
        return image


class LRFreezeImage:
    """
    LRFreezeImage is a class that provides the functionality to freeze an image based on a boolean input.
    If the boolean input 'freeze_image' is True, the input image is saved and passed along. If it is False, the internally saved image is set to None.

    Methods:
        __init__(): Initializes the LRFreezeImage instance and sets up the internal image variable to None.
        
        INPUT_TYPES(): Class method that returns a dictionary specifying the required input types 
                        for the image processing operations. The inputs include image and freeze_image boolean flag.
                        
        process(): Method that takes in the image and freeze_image flag, saves the image if freeze_image is True, 
                    and sets the saved image to None if freeze_image is False.
    """
    def __init__(self):
        self.frozen_image = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
            },
            "optional": {
                "freeze_image": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE",)  
    RETURN_NAMES = ("processed image",)
    FUNCTION = "process"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/visual"

    def process(self, image, freeze_image=None):
        if freeze_image:
            if self.frozen_image is None:
                self.frozen_image = image
            return [self.frozen_image]
        else:
            if self.frozen_image is not None:
                self.frozen_image = None
            return [image]


class LRCropCoordinates():
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("cropping_coordinates", )
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "LunarRing/prompt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "upper": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "right": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lower": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def execute(self, left, upper, right, lower):
        cropping_coordinates = [left, upper, right, lower]
        return (cropping_coordinates, )



class LRCropImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_img": ("IMAGE", {}),
                "cropping_coordinates": ("ARRAY", {})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped image",)
    FUNCTION = "crop"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/visual"

    def crop(self, input_img, cropping_coordinates):
        try:
            if max(cropping_coordinates) <= 1:
                cropping_coordinates = [int(coord * input_img.shape[1]) if i % 2 == 0 else int(coord * input_img.shape[0]) for i, coord in enumerate(cropping_coordinates)]
            pil_image = Image.fromarray(input_img)
            cropped_img = pil_image.crop(cropping_coordinates)
            return [np.array(cropped_img)]
        except Exception as e:
            print(f"Error during cropping: {e}")
            return [input_img]


class LRImageGate:
    def __init__(self):
        self.return_image = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE", {}),
                "inject_image": ("BOOLEAN", {"default": False, "defaultInput": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "gate"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/visual"

    def gate(self, input_image, inject_image=None):
        if inject_image:
            self.return_image = input_image
        return [self.return_image]


class LRImageGateSelect:
    def __init__(self):
        self.return_image = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image_1": ("IMAGE", {}),
                "input_image_2": ("IMAGE", {}),
                "inject_first": ("BOOLEAN", {"default": False, "defaultInput": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "gate"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/visual"

    def gate(self, input_image_1, input_image_2, inject_first=None):
        if inject_first:
            self.return_image = input_image_1
        else:
            self.return_image = input_image_2
        return [self.return_image]
