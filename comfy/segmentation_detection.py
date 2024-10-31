from ..tools.segmentation_detection import HumanSeg, FaceCropper
import numpy as np
from PIL import Image



class LRHumanSeg:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask image",)
    FUNCTION = "get_mask"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/vision"

    def __init__(self):
        self.human_seg = None

    def initialize_once(self, model_name=None, resizing_factor=None, size=None):
        model_name = model_name or 'deeplabv3_resnet101'
        if self.human_seg is None:
            self.human_seg = HumanSeg(model_name=model_name, resizing_factor=resizing_factor, size=size)
            print("Initialized HumanSegNode")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_img": ("IMAGE", {}),
                "model_name": ("STRING", {"default": "deeplabv3_resnet101"}),
                "resizing_factor": ("FLOAT", {"default": 1}),
                # "size": ("TUPLE", {"default": None}),
            }
        }

    def get_mask(self, input_img, model_name=None, resizing_factor=None):
        self.initialize_once(model_name, resizing_factor, size=None)
        print("starting getting mask")
        mask = self.human_seg.get_mask(input_img)
        print("got mask")
        return (mask,)


class LRFaceCropper:
    RETURN_TYPES = ("IMAGE", "ARRAY", "BOOLEAN")
    RETURN_NAMES = ("cropped image", "cropping coordinates", "is_face_present")
    FUNCTION = "crop"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/vision"
    DEFAULT_PADDING = 30

    def __init__(self):
        self.face_cropper = None
        self.last_padding = self.DEFAULT_PADDING

    def initialize_once(self):
        if self.face_cropper is None:
            self.face_cropper = FaceCropper(padding=self.DEFAULT_PADDING)
            print("Initialized FaceCropperNode")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_img": ("IMAGE", {}),
                "padding": ("FLOAT", {"default": 30}),
            }
        }

    def crop(self, input_img, padding=None):
        self.initialize_once()
        # print(f"Padding value: {padding}")
        if padding is not None and self.last_padding != padding:
            self.last_padding = padding
            self.face_cropper.set_padding(int(padding))
        
        cropping_coordinates = self.face_cropper.get_cropping_coordinates(input_img)
        cropped_img = self.face_cropper.apply_crop(input_img, cropping_coordinates)
        if cropping_coordinates:
            cropping_coordinates = np.asarray(cropping_coordinates)
            is_face_present = True
        else:
            is_face_present = False

        
        return (cropped_img, cropping_coordinates, is_face_present)
