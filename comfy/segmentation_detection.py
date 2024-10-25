from tools.segmentation_detection import HumanSeg

class HumanSegNode:
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "get_mask"
    OUTPUT_NODE = False
    CATEGORY = "Segmentation"

    def __init__(self):
        self.human_seg = None

    def initialize_once(self, model_name='deeplabv3_resnet101', resizing_factor=None, size=None):
        if self.human_seg is None:
            self.human_seg = HumanSeg(model_name=model_name, resizing_factor=resizing_factor, size=size)
            print("Initialized HumanSegNode")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_img": ("IMAGE", {}),
                "model_name": ("STRING", {"default": "deeplabv3_resnet101"}),
                "resizing_factor": ("FLOAT", {"default": None}),
                "size": ("TUPLE", {"default": None}),
            }
        }

    def get_mask(self, input_img, model_name='deeplabv3_resnet101', resizing_factor=None, size=None):
        self.initialize_once(model_name, resizing_factor, size)
        mask = self.human_seg.get_mask(input_img)
        return (mask,)
