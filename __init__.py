NODE_CLASS_MAPPINGS = {}
IMPORT_ERROR_MESSAGE = "\n\n\n!!!!!!!!Lunar Ring Real Time Diffusion: failed to import"
try:
    from .comfy.diffusion_engine import LRDiffusionEngineAcid, LRDiffusionEngineLoader
    NODE_CLASS_MAPPINGS["LR DiffusionEngineAcid"] = LRDiffusionEngineAcid
    NODE_CLASS_MAPPINGS["LR DiffusionEngineLoader"] = LRDiffusionEngineLoader
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} diffusion_engine: {e}")

try:
    from .comfy.input_image import LRInputImageProcessor, LRFreezeImage
    NODE_CLASS_MAPPINGS["LR InputImageProcessor"] = LRInputImageProcessor
    NODE_CLASS_MAPPINGS["LR FreezeImage"] = LRFreezeImage
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} input_image: {e}")

try:
    from .comfy.embeddings_mixer import LREncodePrompt, LRBlend2Embeds, LRBlend4Embeds
    NODE_CLASS_MAPPINGS["LR EncodePrompt"] = LREncodePrompt
    NODE_CLASS_MAPPINGS["LR LRBlend2Embeds"] = LRBlend2Embeds
    NODE_CLASS_MAPPINGS["LR LRBlend4Embeds"] = LRBlend4Embeds
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} embeddings_mixer: {e}")