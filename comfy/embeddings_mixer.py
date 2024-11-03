from ..sdxl_turbo.embeddings_mixer import EmbeddingsMixer
import numpy as np

class LRPrompt2Embedding:
    RETURN_TYPES = ("CONDITIONING",)  
    RETURN_NAMES = ("embeds",)  
    FUNCTION = "encode_prompt"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/embeds"
    
    # @classmethod 
    # def IS_CHANGED(self, diffusion_engine, prompt):
    #     if prompt != self.last_prompt:
    #         return 1.0
    #     else:
    #         return 0.0
    
    def __init__(self):
        self.em = None
        self.last_prompt = None

    def initialize_once(self, diffusion_engine):
        if self.em is None:
            self.em = EmbeddingsMixer(diffusion_engine.pipe)
            print("LR: initialize_once LREncodePrompt")
            
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {       
                    "diffusion_engine": ("MODEL", {}),
                    "prompt": ("STRING", {"multiline": False, "default": ""}),
                    }
                }

    def encode_prompt(self, diffusion_engine, prompt):
        self.initialize_once(diffusion_engine)
        if self.last_prompt != prompt:
            print(f'encode prompt is called with: {prompt}')
            self.last_embeds = [self.em.encode_prompt(prompt)]
            self.last_prompt = prompt
        return (self.last_embeds)


class LRBlend2Embeds:
    RETURN_TYPES = ("CONDITIONING",)  
    RETURN_NAMES = ("embeds",)  
    FUNCTION = "blend"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/embeds"

    def __init__(self):
        self.em = None

    def initialize_once(self, diffusion_engine):
        if self.em is None:
            self.em = EmbeddingsMixer(diffusion_engine.pipe)
            print("LR: initialize_once LREncodePrompt")
            
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {       
                    "diffusion_engine": ("MODEL", {}),
                    "embeds1": ("CONDITIONING", {}),
                    "embeds2": ("CONDITIONING", {}),
                    "weight": ("FLOAT", {
                        "default": 0.5, 
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "display": "number"
                        }),
                    }
                }

    def blend(self, diffusion_engine, embeds1, embeds2, weight):
        self.initialize_once(diffusion_engine)
        embeds = [self.em.blend_two_embeds(embeds1, embeds2, weight)]
        return (embeds)
    

class LRBlend4Embeds:
    RETURN_TYPES = ("CONDITIONING",)  
    RETURN_NAMES = ("embeds",)  
    FUNCTION = "blend"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/embeds"

    def __init__(self):
        self.em = None

    def initialize_once(self, diffusion_engine):
        if self.em is None:
            self.em = EmbeddingsMixer(diffusion_engine.pipe)
            print("LR: initialize_once LREncodePrompt")
            
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {       
                    "diffusion_engine": ("MODEL", {}),
                    "embeds1": ("CONDITIONING", {}),
                    "weight1": ("FLOAT", {
                        "default": 0.25, 
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "display": "number"
                        }),
                    "embeds2": ("CONDITIONING", {}),
                    "weight2": ("FLOAT", {
                        "default": 0.25, 
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "display": "number"
                        }),
                    "embeds3": ("CONDITIONING", {}),
                    "weight3": ("FLOAT", {
                        "default": 0.25, 
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "display": "number"
                        }),
                    "embeds4": ("CONDITIONING", {}),
                    "weight4": ("FLOAT", {
                        "default": 0.25, 
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "display": "number"
                        }),
                    }
                }

    def blend(self, diffusion_engine, embeds1=None, weight1=1, embeds2=None, weight2=1, embeds3=None, weight3=1, embeds4=None, weight4=1):
        self.initialize_once(diffusion_engine)
        list_weights = [weight for weight, embed in [(weight1, embeds1), (weight2, embeds2), (weight3, embeds3), (weight4, embeds4)] if embed is not None]
        list_embeds = [embed for embed in [embeds1, embeds2, embeds3, embeds4] if embed is not None]
        embeds = [self.em.blend_multi_embeds(list_embeds, list_weights)]
        return (embeds)

# # Add custom API routes, using router
# from aiohttp import web
# from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")

