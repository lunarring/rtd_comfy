NODE_CLASS_MAPPINGS = {}
from .comfy.diffusion_engine import LRDiffusionEngineAcid, LRDiffusionEngineLoader
NODE_CLASS_MAPPINGS["LR DiffusionEngineAcid"] = LRDiffusionEngineAcid
NODE_CLASS_MAPPINGS["LR DiffusionEngineLoader"] = LRDiffusionEngineLoader

from .comfy.input_image import LRInputImageProcessor, LRFreezeImage
NODE_CLASS_MAPPINGS["LR InputImageProcessor"] = LRInputImageProcessor
NODE_CLASS_MAPPINGS["LR FreezeImage"] = LRFreezeImage

from .comfy.embeddings_mixer import LREncodePrompt, LRBlend2Embeds, LRBlend4Embeds
NODE_CLASS_MAPPINGS["LR EncodePrompt"] = LREncodePrompt
NODE_CLASS_MAPPINGS["LR LRBlend2Embeds"] = LRBlend2Embeds
NODE_CLASS_MAPPINGS["LR LRBlend4Embeds"] = LRBlend4Embeds

from .comfy.segmentation_detection import LRHumanSeg
NODE_CLASS_MAPPINGS["LR HumanSeg"] = LRHumanSeg
