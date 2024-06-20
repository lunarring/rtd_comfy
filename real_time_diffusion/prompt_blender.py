#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:30:17 2023

@author: lunar
"""

import torch
import numpy as np
import random
import lunar_tools as lt
import time

@staticmethod
@torch.no_grad()
#%%
class PromptBlender:
    """
    The PromptBlender class is used to blend two prompts together using spherical interpolation and image similarity metrics.
    It uses the LPIPS (Learned Perceptual Image Patch Similarity) metric to compute the similarity between two images.
    The class also contains helper functions to retrieve the parents for any given mixing and to compute which parental latents should be mixed together to achieve a smooth blend.
    """
    def __init__(self, pipe, gpu_id=0):
        self.pipe = pipe
        self.fract = 0
        self.first_fract = 0
        self.gpu_id = gpu_id
        self.embeds1 = None
        self.embeds2 = None
        self.embeds_current = None
        self.device = "cuda"
        self.tree_final_imgs = None
        self.tree_fracts = None
        self.tree_similarities = None
        self.tree_insertion_idx = None

    def set_prompt1(self, prompt, negative_prompt=""):
        self.embeds1 = self.get_prompt_embeds(prompt, negative_prompt)
        if self.embeds_current is None:
            self.embeds_current = self.embeds1 
    
    def set_prompt2(self, prompt, negative_prompt=""):
        self.embeds2 = self.get_prompt_embeds(prompt, negative_prompt)


    def get_prompt_embeds(self, prompt, negative_prompt=""):
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
            device=f"cuda:{self.gpu_id}",
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

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    

    def blend_stored_embeddings(self, fract):
        """
        Blends the stored embeddings `embeds1` and `embeds2` based on the given fraction `fract`.

        This function ensures that both `embeds1` and `embeds2` are set before proceeding. It then
        clips the fraction to be within the range [0.0, 1.0] and uses the `blend_prompts` method to
        blend the embeddings. The blended embeddings are stored in `embeds_current` and returned.

        Args:
            fract (float): The fraction to blend the embeddings. Should be between 0.0 and 1.0.

        Returns:
            tuple: A tuple containing the blended prompt embeddings, negative prompt embeddings,
                   pooled prompt embeddings, and negative pooled prompt embeddings.
        """
        assert hasattr(self, 'embeds1'), "embeds1 not set. Please set embeds1 before blending."
        assert hasattr(self, 'embeds2'), "embeds2 not set. Please set embeds2 before blending."
        fract = np.clip(fract, 0.0, 1.0)
        self.embeds_current = self.blend_prompts(self.embeds1, self.embeds2, fract)
        self.prompt_embeds, self.negative_prompt_embeds, self.pooled_prompt_embeds, self.negative_pooled_prompt_embeds = self.embeds_current
        return self.prompt_embeds, self.negative_prompt_embeds, self.pooled_prompt_embeds, self.negative_pooled_prompt_embeds 


    def blend_prompts(self, embeds1, embeds2, fract):
        """
        Blends two sets of prompt embeddings based on a specified fraction.
        """
        prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1 = embeds1
        prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2 = embeds2

        blended_prompt_embeds = self.interpolate_spherical(prompt_embeds1, prompt_embeds2, fract)
        blended_negative_prompt_embeds = self.interpolate_spherical(negative_prompt_embeds1, negative_prompt_embeds2, fract)
        blended_pooled_prompt_embeds = self.interpolate_spherical(pooled_prompt_embeds1, pooled_prompt_embeds2, fract)
        blended_negative_pooled_prompt_embeds = self.interpolate_spherical(negative_pooled_prompt_embeds1, negative_pooled_prompt_embeds2, fract)

        return blended_prompt_embeds, blended_negative_prompt_embeds, blended_pooled_prompt_embeds, blended_negative_pooled_prompt_embeds


    def get_all_embeddings(self, list_prompts):
        prompts_embeds = []
        for prompt in list_prompts:
            prompts_embeds.append(self.get_prompt_embeds(prompt))
        self.prompts_embeds = prompts_embeds
    
#% Linear Walker (legacy)
    def blend_sequence_prompts(self, prompts, n_steps):
        """
        Generates a sequence of blended prompt embeddings for a list of text prompts.
        """
        blended_prompts = []
        for i in range(len(prompts) - 1):
            prompt_embeds1 = self.get_prompt_embeds(prompts[i])
            prompt_embeds2 = self.get_prompt_embeds(prompts[i + 1])
            for step in range(n_steps):
                fract = step / float(n_steps - 1)
                blended = self.blend_prompts(prompt_embeds1, prompt_embeds2, fract)
                blended_prompts.append(blended)
        return blended_prompts
    
    
    def set_init_position(self, index):
        self.current = [self.prompts_embeds[index][i] for i in range(4)]
    
    def set_target(self, index):
        self.target = [self.prompts_embeds[index][i] for i in range(4)]
    
    def step(self, pvelocity):
        for i in range(4):
            d = self.target[i] - self.current[i]
            
            d_norm = torch.linalg.norm(d)
            if d_norm > 0:
                # self.fract = pvelocity / d_norm
                # self.fract = torch.sqrt(self.fract)
                self.fract = pvelocity
                
                # self.fract = pvelocity
                if self.fract > 1:
                    self.fract = 1
            else:
                self.fract = 1
            
            self.current[i] = self.interpolate_spherical(self.current[i], self.target[i], self.fract)
            if i == 0:
                self.first_fract = self.fract

    # TREE (latent blending lpips fract patterning)
    def init_tree(self, img_first=None, img_last=None, latents=None):
        if img_first is None:
            img_first = self.generate_blended_img(0.0, latents)
        if img_last is None:
            img_last = self.generate_blended_img(1.0, latents)
        self.tree_final_imgs = [img_first, img_last]
        self.tree_fracts = [0.0, 1.0]
        self.tree_similarities = [self.get_lpips_similarity(img_first, img_last)]
        self.tree_insertion_idx = [0, 0]


    def insert_into_tree(self, img_insert, fract_mixing):
        r"""
        Inserts all necessary parameters into the trajectory tree.
        Args:
            fract_mixing: float
                the fraction along the transition axis [0, 1]
        """
        
        b_parent1, b_parent2 = self.get_closest_idx(fract_mixing)
        left_sim = self.get_lpips_similarity(img_insert, self.tree_final_imgs[b_parent1])
        right_sim = self.get_lpips_similarity(img_insert, self.tree_final_imgs[b_parent2])
        idx_insert = b_parent1 + 1
        self.tree_final_imgs.insert(idx_insert, img_insert)
        self.tree_fracts.insert(idx_insert, fract_mixing)
        idx_max = np.max(self.tree_insertion_idx) + 1
        self.tree_insertion_idx.insert(idx_insert, idx_max)
        
        # update similarities
        self.tree_similarities[b_parent1] = left_sim
        self.tree_similarities.insert(idx_insert, right_sim)


    # Auxiliary functions
    def get_closest_idx(
            self,
            fract_mixing: float):
        r"""
        Helper function to retrieve the parents for any given mixing.
        Example: fract_mixing = 0.4 and self.tree_fracts = [0, 0.3, 0.6, 1.0]
        Will return the two closest values here, i.e. [1, 2]
        """

        pdist = fract_mixing - np.asarray(self.tree_fracts)
        pdist_pos = pdist.copy()
        pdist_pos[pdist_pos < 0] = np.inf
        b_parent1 = np.argmin(pdist_pos)
        pdist_neg = -pdist.copy()
        pdist_neg[pdist_neg <= 0] = np.inf
        b_parent2 = np.argmin(pdist_neg)

        if b_parent1 > b_parent2:
            tmp = b_parent2
            b_parent2 = b_parent1
            b_parent1 = tmp

        return b_parent1, b_parent2

    def get_mixing_parameters(self):
        r"""
        Computes which parental latents should be mixed together to achieve a smooth blend.
        As metric, we are using lpips image similarity. The insertion takes place
        where the metric is maximal.
        """
        # get_lpips_similarity
        similarities = self.tree_similarities
        # similarities = self.get_tree_similarities()
        b_closest1 = np.argmax(similarities)
        b_closest2 = b_closest1 + 1
        fract_closest1 = self.tree_fracts[b_closest1]
        fract_closest2 = self.tree_fracts[b_closest2]
        fract_mixing = (fract_closest1 + fract_closest2) / 2

        return fract_mixing, b_closest1, b_closest2

    
    def load_lpips(self):
        import lpips
        self.lpips = lpips.LPIPS(net='alex').cuda(self.gpu_id)

    def get_lpips_similarity(self, imgA, imgB):
        r"""
        Computes the image similarity between two images imgA and imgB.
        Used to determine the optimal point of insertion to create smooth transitions.
        High values indicate low similarity.
        """
        tensorA = torch.from_numpy(np.asarray(imgA)).float().cuda(self.device)
        tensorA = 2 * tensorA / 255.0 - 1
        tensorA = tensorA.permute([2, 0, 1]).unsqueeze(0)
        tensorB = torch.from_numpy(np.asarray(imgB)).float().cuda(self.device)
        tensorB = 2 * tensorB / 255.0 - 1
        tensorB = tensorB.permute([2, 0, 1]).unsqueeze(0)
        lploss = self.lpips(tensorA, tensorB)
        lploss = float(lploss[0][0][0][0])
        return lploss
        
    def interpolate_spherical(self, p0, p1, fract_mixing: float):
        """
        Helper function to correctly mix two random variables using spherical interpolation.
        """
        if p0.dtype == torch.float16:
            recast_to = 'fp16'
        else:
            recast_to = 'fp32'

        p0 = p0.double()
        p1 = p1.double()
        norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
        epsilon = 1e-7
        dot = torch.sum(p0 * p1) / norm
        dot = dot.clamp(-1 + epsilon, 1 - epsilon)

        theta_0 = torch.arccos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * fract_mixing
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = torch.sin(theta_t) / sin_theta_0
        interp = p0 * s0 + p1 * s1

        if recast_to == 'fp16':
            interp = interp.half()
        elif recast_to == 'fp32':
            interp = interp.float()

        return interp


    
