NODE_CLASS_MAPPINGS = {}
from .comfy.diffusion_engine import LRDiffusionEngineAcid, LRDiffusionEngineLoader, LRDiffusionEngineThreaded
NODE_CLASS_MAPPINGS["LR DiffusionEngineAcid"] = LRDiffusionEngineAcid
NODE_CLASS_MAPPINGS["LR DiffusionEngineLoader"] = LRDiffusionEngineLoader
NODE_CLASS_MAPPINGS["LR DiffusionEngineThreaded"] = LRDiffusionEngineThreaded

from .comfy.input_image import LRInputImageProcessor, LRFreezeImage, LRCropImage, LRImageGate, LRImageGateSelect, LRCropCoordinates
NODE_CLASS_MAPPINGS["LR InputImageProcessor"] = LRInputImageProcessor
NODE_CLASS_MAPPINGS["LR FreezeImage"] = LRFreezeImage
NODE_CLASS_MAPPINGS["LR CropImage"] = LRCropImage
NODE_CLASS_MAPPINGS["LR CropCoordinates"] = LRCropCoordinates
NODE_CLASS_MAPPINGS["LR ImageGate"] = LRImageGate
NODE_CLASS_MAPPINGS["LR ImageGateSelect"] = LRImageGateSelect

from .comfy.embeddings_mixer import LRPrompt2Embedding, LRBlend2Embeds, LRBlend4Embeds
NODE_CLASS_MAPPINGS["LR Prompt2Embedding"] = LRPrompt2Embedding
NODE_CLASS_MAPPINGS["LR LRBlend2Embeds"] = LRBlend2Embeds
NODE_CLASS_MAPPINGS["LR LRBlend4Embeds"] = LRBlend4Embeds

from .comfy.segmentation_detection import LRHumanSeg, LRFaceCropper
NODE_CLASS_MAPPINGS["LR HumanSeg"] = LRHumanSeg
NODE_CLASS_MAPPINGS["LR FaceCropper"] = LRFaceCropper
