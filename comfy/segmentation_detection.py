from ..tools.segmentation_detection import HumanSeg

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
        model_name = model_name or 'deeplabv3_resnet101'
        self.initialize_once(model_name, resizing_factor, size=None)
        mask = self.human_seg.get_mask(input_img)
        return (mask,)

