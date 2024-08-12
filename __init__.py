NODE_CLASS_MAPPINGS = {}
IMPORT_ERROR_MESSAGE = "\n\n\n!!!!!!!!!!!!Lunar Ring Real Time Diffusion: failed to import"
try:
    from .comfy.diffusion_engine import LRDiffusionEngineAcid, LRDiffusionEngineLoader
    NODE_CLASS_MAPPINGS["LR DiffusionEngineAcid"] = LRDiffusionEngineAcid
    NODE_CLASS_MAPPINGS["LR DiffusionEngineLoader"] = LRDiffusionEngineLoader
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE}: {e}")