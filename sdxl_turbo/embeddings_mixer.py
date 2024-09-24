#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import lunar_tools as lt
from diffusers import AutoPipelineForText2Image

@staticmethod
@torch.no_grad()
class EmbeddingsMixer:
    """
    The EmbedMixer class is used to encode text prompts into embeddings using a given model pipeline. 
    It provides methods to set a list of prompts and add a new prompt to the existing list. 
    The encoded embeddings can be used for further processing in the model pipeline.

    Attributes:
        pipe: The model pipeline used for encoding.
        device: The device on which the computations will be performed.
        prompts_embeds: A list of encoded embeddings for the prompts.
        list_prompts: A list of text prompts to be encoded.
    """
    def __init__(self, pipe):
        self.pipe = pipe
        self.device = str(pipe._execution_device)
        self.prompts_embeds = []
        self.list_prompts = []

    @torch.no_grad()
    def encode_prompt(self, prompt, negative_prompt=""):
        """
        Encodes a text prompt into embeddings using the model pipeline.
        """
        (
         prompt_embeds, 
         negative_prompt_embeds, 
         pooled_prompt_embeds, 
         negative_pooled_prompt_embeds
         ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=0,
            clip_skip=False
        )
        embeds = [prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds]
        return embeds
    
    @torch.no_grad()
    def encode_and_store_prompts(self, list_prompts):
        """
        Sets the list of prompts and encodes them into embeddings.

        Args:
            list_prompts (list): A list of text prompts to be encoded.

        """
        self.prompts_embeds = []
        self.list_prompts = list_prompts
        for prompt in self.list_prompts:
            self.prompts_embeds.append(self.encode_prompt(prompt))


    def add_prompt(self, prompt):
        """
        Adds a new prompt to the list of prompts and encodes it into embeddings.

        Args:
            prompt (str): The text prompt to be added and encoded.
        """
        embeds = self.encode_prompt(prompt)
        self.prompts_embeds.append(embeds)
        self.list_prompts.append(prompt)
        self.prompts_embeds.append(self.encode_prompt(prompt))

    def blend_two_prompts(self, prompt1, prompt2, weight):
        """
        Blends two prompts by encoding them into embeddings and then blending the embeddings.

        Args:
            prompt1 (str): The first text prompt to blend.
            prompt2 (str): The second text prompt to blend.
            weight (float): The weight for the blending. A value of 0 returns the embeddings of `prompt1`, a value of 1 returns the embeddings of `prompt2`.

        Returns:
            list: The blended embeddings. It is a list of four tensors: 
                  blended_prompt_embeds, blended_negative_prompt_embeds, blended_pooled_prompt_embeds, blended_negative_pooled_prompt_embeds.
        """
        embeds1 = self.encode_prompt(prompt1)
        embeds2 = self.encode_prompt(prompt2)
        return self.blend_two_embeds(embeds1, embeds2, weight)


    def blend_two_embeds(self, embeds1, embeds2, weight):
        """
        Blends two sets of embeddings with a given weight.

        Args:
            embeds1 (list): The first set of embeddings to blend. It is a list of four tensors: 
                            prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1.
            embeds2 (list): The second set of embeddings to blend. It is a list of four tensors: 
                            prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2.
            weight (float): The weight for the blending. A value of 0 returns `embeds1`, a value of 1 returns `embeds2`.

        Returns:
            list: The blended embeddings. It is a list of four tensors: 
                  blended_prompt_embeds, blended_negative_prompt_embeds, blended_pooled_prompt_embeds, blended_negative_pooled_prompt_embeds.
        """
        blended_embeds = self.blend_multi_embeds([embeds1, embeds2], [1-weight, weight])
        return blended_embeds


    def blend_multi_embeds(self, list_embeds, weights):
        """
        Blends multiple sets of embeddings with given weights.

        Args:
            list_embeds (list): A list of sets of embeddings to blend. Each set of embeddings is a list of four tensors: 
                                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds.
            weights (list or np.ndarray): A list or numpy array of weights for the blending. Each weight corresponds to a set of embeddings in `list_embeds`.

        Returns:
            list: The blended embeddings. It is a list of four tensors: 
                  blended_prompt_embeds, blended_negative_prompt_embeds, blended_pooled_prompt_embeds, blended_negative_pooled_prompt_embeds.

        Note:
            The tensors in each set of embeddings must have the same shape, and the weights must be a list of scalars or a numpy array.
            If weights is a numpy array, it is converted to a list.
            The length of `list_embeds` and `weights` must be the same.
        """
        if isinstance(weights, np.ndarray):
            weights = weights.tolist()
        assert len(list_embeds) == len(weights), "The length of list_embeds and weights must be the same"

        embeds_mixed = []
        nmb_dim = len(list_embeds[0])
        for i in range(nmb_dim):
            for j in range(len(weights)):
                if j==0:
                    emb = list_embeds[j][i] * weights[j]
                else:
                    emb += list_embeds[j][i] * weights[j]
            embeds_mixed.append(emb)
        return embeds_mixed


    def blend_stored_embeddings(self, weights_embeds, idx_embeds=None):
        """
        Blends stored embeddings based on given indices and weights, e.g. the embeddings set by encode_and_store_prompts().

        Args:
            weights_embeds (list or float): A list of weights for the blending. Each weight corresponds to an embedding in `idx_embeds`.
                                          If `idx_embeds` has length 2, `weights_embeds` can be a single float.
            idx_embeds (list): A list of indices corresponding to the stored embeddings to blend.

        Returns:
            list: The blended embeddings. It is a list of four tensors: 
                  blended_prompt_embeds, blended_negative_prompt_embeds, blended_pooled_prompt_embeds, blended_negative_pooled_prompt_embeds.

        Note:
            The tensors in each set of embeddings must have the same shape, and the weights must be a list of scalars.
            The length of `idx_embeds` and `weights_embeds` must be the same.
        """
        if idx_embeds is None:
            idx_embeds = list(range(len(self.prompts_embeds)))
        assert all(idx < len(self.prompts_embeds) for idx in idx_embeds), "Index out of range in self.prompts_embeds"
        # assert len(idx_embeds) >= 2, "The length of idx_embeds should be at least 2"
        if len(idx_embeds) == 2:
            if isinstance(weights_embeds, list):
                assert len(weights_embeds) == 1, "weights_embeds should have length 1"
                weights_embeds = weights_embeds[0]
            embeds_mixed = self.blend_two_embeds(self.prompts_embeds[idx_embeds[0]], self.prompts_embeds[idx_embeds[1]], weights_embeds)
        else:
            assert isinstance(weights_embeds, list), "weights_embeds should be a list"
            assert len(weights_embeds) == len(idx_embeds), "The length of weights_embeds should be equal to the length of idx_embeds"
            list_embeds = [self.prompts_embeds[idx] for idx in idx_embeds]
            embeds_mixed = self.blend_multi_embeds(list_embeds, weights_embeds)
        return embeds_mixed

if __name__ == "__main__":
    from diffusion_engine import DiffusionEngine
#%%
    de_txt = DiffusionEngine(use_image2image=False, height_diffusion_desired=512, width_diffusion_desired=512)
    em = EmbeddingsMixer(de_txt.pipe)
    embeds = em.encode_prompt("photo of a house", "blue")

    height_diffusion = 512
    width_diffusion = 512
    de_txt.set_embeddings(embeds)
    img = de_txt.generate()

"""
tensor([[[-3.8926, -2.5117,  4.7109,  ...,  0.1897,  0.4185, -0.2969],
         [-0.4678, -1.5508,  0.0325,  ...,  0.3672,  0.1084,  0.2808],
         [ 0.4060, -0.7217, -0.3323,  

"""
#%%
    # em = EmbeddingsMixer(pipe)

    # list_prompts = []
    # list_prompts.append("beautiful painting of a medieval temple")
    # list_prompts.append("abstract expressionist painting of a landscape on the moon, blue colors")
    # list_prompts.append("photo of a metal statue of a bird")
    # list_prompts.append("photo of a computer screen with a angry error message")
    # em.encode_and_store_prompts(list_prompts)
    
    # idx_embeds = [1,2,3]
    # weights_embeds = [0.9, 0.8, 1.1]
    
    # p_emb, neg_p_emb, pool_p_emb, neg_pool_p_emb = em.blend_stored_embeddings(weights_embeds, idx_embeds)
    # img = pipe(num_inference_steps=1, guidance_scale=0, prompt_embeds=p_emb, negative_prompt_embeds=neg_p_emb, pooled_prompt_embeds=pool_p_emb, negative_pooled_prompt_embeds=neg_pool_p_emb)[0][0]
    # img.show()
    
