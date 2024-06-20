#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%`
import sys
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
import torch
import time

from diffusers import AutoencoderTiny

from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
from diffusers.utils import load_image
import random
import xformers
import triton
import lunar_tools as lt
from PIL import Image
import numpy as np
from diffusers.utils.torch_utils import randn_tensor

import numpy as np
import xformers
import triton
import cv2
import sys
from datasets import load_dataset
from prompt_blender import PromptBlender
import os
from dotenv import load_dotenv #pip install python-dotenv
from kornia.filters.kernels import get_binary_kernel2d
from typing import List, Tuple
from human_seg import HumanSeg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False 

import matplotlib.pyplot as plt
import cv2

#%% VARS
shape_cam = (300,400)
do_compile = True

sz_renderwin = (int(512*2.09), int(512*3.85))
resolution_factor = 8
base_w = 12
base_h = 16

negative_prompt = "blurry, bland, black and white, monochromatic"
model_turbo = "stabilityai/sdxl-turbo"
model_vae = "madebyollin/taesdxl"


#%% These should live in lunar tools
def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]

def median_blur(input: torch.Tensor,
                kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Blurs an image using the median filter.

    Args:
        input (torch.Tensor): the input image with shape :math:`(B,C,H,W)`.
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor with shape :math:`(B,C,H,W)`.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> output = median_blur(input, (3, 3))
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel: torch.Tensor = get_binary_kernel2d(kernel_size).to(input)
    b, c, h, w = input.shape

    # map the local window to single vector
    features: torch.Tensor = F.conv2d(
        input.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1)
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=2)[0]

    return median

class MedianBlur(nn.Module):
    r"""Blurs an image using the median filter.

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = MedianBlur((3, 3))
        >>> output = blur(input)
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """

    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super(MedianBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return median_blur(input, self.kernel_size)

def zoom_image_torch(input_tensor, zoom_factor):
    # Ensure the input is a 4D tensor [batch_size, channels, height, width]
    input_tensor = input_tensor.permute(2,0,1)
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    # Original size
    original_height, original_width = input_tensor.shape[2], input_tensor.shape[3]
    
    # Calculate new size
    new_height = int(original_height * zoom_factor)
    new_width = int(original_width * zoom_factor)
    
    # Interpolate
    zoomed_tensor = F.interpolate(input_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
    # zoomed_tensor = F.interpolate(input_tensor, size=(new_width, new_height), mode='bilinear', align_corners=False).permute(1,0,2)
    
    # Calculate padding to match original size
    pad_height = (original_height - new_height) // 2
    pad_width = (original_width - new_width) // 2
    
    # Adjust for even dimensions to avoid negative padding
    pad_height_extra = original_height - new_height - 2*pad_height
    pad_width_extra = original_width - new_width - 2*pad_width
    
    # Pad to original size
    if zoom_factor < 1:
        zoomed_tensor = F.pad(zoomed_tensor, (pad_width, pad_width + pad_width_extra, pad_height, pad_height + pad_height_extra), 'reflect', 0)
    else:
        # For zoom_factor > 1, center crop to original dimensions
        start_row = (zoomed_tensor.shape[2] - original_height) // 2
        start_col = (zoomed_tensor.shape[3] - original_width) // 2
        zoomed_tensor = zoomed_tensor[:, :, start_row:start_row + original_height, start_col:start_col + original_width]
    
    return zoomed_tensor.squeeze(0).permute(1,2,0)  # Remove batch dimension before returning

def ten2img(ten):
    return ten.cpu().numpy().astype(np.uint8)

def multi_match_gpu(list_images, weights=None, simple=False, clip_max='auto', gpu=0,  is_input_tensor=False):
    """
    Match colors of images according to weights.
    """
    from scipy import linalg
    if is_input_tensor:
        list_images_gpu = [img.clone() for img in list_images]
    else:
        list_images_gpu = [torch.from_numpy(img.copy()).float().cuda(gpu) for img in list_images]
    
    if clip_max == 'auto':
        clip_max = 255 if list_images[0].max() > 16 else 1  
    
    if weights is None:
        weights = [1]*len(list_images_gpu)
    weights = np.array(weights, dtype=np.float32)/sum(weights) 
    assert len(weights) == len(list_images_gpu)
    # try:
    assert simple == False    
    def cov_colors(img):
        a, b, c = img.size()
        img_reshaped = img.view(a*b,c)
        mu = torch.mean(img_reshaped, 0, keepdim=True)
        img_reshaped -= mu
        cov = torch.mm(img_reshaped.t(), img_reshaped) / img_reshaped.shape[0]
        return cov, mu
    
    covs = np.zeros((len(list_images_gpu),3,3), dtype=np.float32)
    mus = torch.zeros((len(list_images_gpu),3)).float().cuda(gpu)
    mu_target = torch.zeros((1,1,3)).float().cuda(gpu)
    #cov_target = np.zeros((3,3), dtype=np.float32)
    for i, img in enumerate(list_images_gpu):
        cov, mu = cov_colors(img)
        mus[i,:] = mu
        covs[i,:,:]= cov.cpu().numpy()
        mu_target += mu * weights[i]
            
    cov_target = np.sum(weights.reshape(-1,1,1)*covs, 0)
    covs += np.eye(3, dtype=np.float32)*1
    
    # inversion_fail = False
    try:
        sqrtK = linalg.sqrtm(cov_target)
        assert np.isnan(sqrtK.mean()) == False
    except Exception as e:
        print(e)
        # inversion_fail = True
        sqrtK = linalg.sqrtm(cov_target + np.random.rand(3,3)*0.01)
    list_images_new = []
    for i, img in enumerate(list_images_gpu):
        
        Ms = np.real(np.matmul(sqrtK, linalg.inv(linalg.sqrtm(covs[i]))))
        Ms = torch.from_numpy(Ms).float().cuda(gpu)
        #img_new = img - mus[i]
        img_new = torch.mm(img.view([img.shape[0]*img.shape[1],3]), Ms.t())
        img_new = img_new.view([img.shape[0],img.shape[1],3]) + mu_target
        
        img_new = torch.clamp(img_new, 0, clip_max)

        assert torch.isnan(img_new).max().item() == False
        if is_input_tensor:
            list_images_new.append(img_new)
        else:
            list_images_new.append(img_new.cpu().numpy())
    return list_images_new
#%% Inits
cam = lt.WebCam(cam_id=-1, shape_hw=shape_cam)
cam.autofocus_enable()

# Diffusion Pipe
pipe = AutoPipelineForImage2Image.from_pretrained(model_turbo, torch_dtype=torch.float16, variant="fp16", local_files_only=False)
pipe.to("cuda")
pipe.vae = AutoencoderTiny.from_pretrained(model_vae, torch_device='cuda', torch_dtype=torch.float16, local_files_only=False)
pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)

if do_compile:
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    config.enable_jit = True
    config.enable_jit_freeze = True
    config.trace_scheduler = True
    config.enable_cnn_optimization = True
    config.preserve_parameters = False
    config.prefer_lowp_gemm = True
    pipe = compile(pipe, config)
    
# human body segmentation
human_seg = HumanSeg()

# Promptblender
blender = PromptBlender(pipe)

meta_input = lt.MetaInput()
speech_detector = lt.Speech2Text()

renderer = lt.Renderer(width=sz_renderwin[1], height=sz_renderwin[0])
blur = MedianBlur((7, 7))

#%%
prompt = 'Bizarre creature from Hieronymus Bosch painting "A Garden of Earthly Delights" on a schizophrenic ayahuasca trip'

fract = 0
blender.set_prompt1(prompt)
blender.set_prompt2(prompt)
prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.blend_stored_embeddings(fract)
blender.blend_stored_embeddings(fract)

# Renderer
cam_img = cam.get_img()
cam_img = np.flip(cam_img, axis=1)

noise_resolution_w = base_w*resolution_factor
noise_resolution_h = base_h*resolution_factor

cam_resolution_w = base_w*8*resolution_factor
cam_resolution_h = base_h*8*resolution_factor

# test resolution
cam_img = cv2.resize(cam_img.astype(np.uint8), (cam_resolution_w, cam_resolution_h))
last_cam_img_torch = None


# noise
latents = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda()
noise_img2img_orig = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda()

torch_last_diffusion_image = torch.from_numpy(cam_img).to(latents.device, dtype=torch.float)
image_displacement_accumulated = 0
image_displacement_array_accumulated = None


#%% LOOP

guidance_scale = 0.5
num_inference_steps = 2
strength = 0.5
t_last_frame_rendered = time.time()
fract = 0
movie_recording_started = 0

while True:
    torch.manual_seed(0)
    noise_img2img_fresh = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda()
    
    noise_mixing = meta_input.get(akai_midimix="D0", val_min=0, val_max=1.0, val_default=0)
    noise_img2img = blender.interpolate_spherical(noise_img2img_orig, noise_img2img_fresh, noise_mixing)

    do_record_mic = False
    
    if do_record_mic:
        if not speech_detector.audio_recorder.is_recording:
            speech_detector.start_recording()
    elif not do_record_mic:
        if speech_detector.audio_recorder.is_recording:
            try:
                prompt_prev = prompt
                prompt = speech_detector.stop_recording()
                print(f"New prompt: {prompt}")
                stop_recording = False
                fract = 0
                blender.set_prompt1(prompt_prev, negative_prompt)
                blender.set_prompt2(prompt, negative_prompt)
                
            except Exception as e:
                print(f"FAIL {e}")
            
    get_new_prompt = meta_input.get(akai_midimix='B3', akai_lpd8='A0', button_mode='pressed_once')
    if get_new_prompt:
        try:
            prompt_prev = prompt
            print(f"New prompt: {prompt}")
            fract = 0
            blender.set_prompt1(prompt_prev, negative_prompt)
            blender.set_prompt2(prompt, negative_prompt)
        except Exception as e:
            print(f"fail! {e}")
            
    
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.blend_stored_embeddings(fract)


    cam_img = cam.get_img()
    cam_img = np.flip(cam_img, axis=1)
    
    # mask the body
    apply_body_mask = meta_input.get(akai_midimix="E3", button_mode="toggle", val_default=True)
    if apply_body_mask:
        mask_strength = meta_input.get(akai_midimix="E2", val_min=0.0, val_max=1.0, val_default=1)
        mask = np.expand_dims(human_seg.get_mask(cam_img), axis=2)
        cam_img_masked = cam_img * mask
        cam_img = (1-mask_strength)*cam_img + mask_strength * cam_img_masked
        cam_img = cam_img.astype(np.uint8)
    
    cam_img = cv2.resize(cam_img.astype(np.uint8), (cam_resolution_w, cam_resolution_h))
    
    
    cam_img_torch = torch.from_numpy(cam_img.copy()).to(latents.device).float()
    cam_img_torch = blur(cam_img_torch.permute([2,0,1])[None])[0].permute([1,2,0])
    
    # coef noise
    coef_noise = meta_input.get(akai_midimix="E0", akai_lpd8="E1", val_min=0, val_max=0.5, val_default=0.15)
    
    t_rand = (torch.randn(cam_img_torch.shape[0], cam_img_torch.shape[1], 3, device=cam_img_torch.device) - 0.5) * coef_noise * 255

    cam_img_torch += t_rand
    torch_last_diffusion_image += t_rand
    
    acid_strength = meta_input.get(akai_midimix="C0", akai_lpd8="F0", val_min=0, val_max=0.8, val_default=0.11)
    acid_freq = meta_input.get(akai_midimix="F2", val_min=0, val_max=10.0, val_default=0)

    cam_img_torch = (1.-acid_strength)*cam_img_torch + acid_strength*torch_last_diffusion_image
    cam_img_torch = torch.clamp(cam_img_torch, 0, 255)


    cam_img = cam_img_torch.cpu().numpy()

    use_debug_overlay = meta_input.get(akai_midimix="H3", akai_lpd8="D1", button_mode="toggle")
    if use_debug_overlay:
        image = cam_img.astype(np.uint8)
    else:
        image = pipe(image=Image.fromarray(cam_img.astype(np.uint8)), 
                      latents=latents, num_inference_steps=num_inference_steps, strength=0.5, 
                      guidance_scale=guidance_scale, prompt_embeds=prompt_embeds, 
                      negative_prompt_embeds=negative_prompt_embeds, 
                      pooled_prompt_embeds=pooled_prompt_embeds, 
                      negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, noise_img2img=noise_img2img).images[0]
        
    dt_frame = time.time() - t_last_frame_rendered
    t_last_frame_rendered = time.time()
    
    fps = np.round(1/dt_frame)
    lt.dynamic_print(f'fps: {fps}')
    try:
        torch_last_diffusion_image = torchvision.transforms.functional.pil_to_tensor(image).to(latents.device, dtype=torch.float).permute(1,2,0)
    except:
        torch_last_diffusion_image = torch.from_numpy(image).to(latents.device, dtype=torch.float)

    # Render the image
    renderer.render(image)
    
    do_record_movie = meta_input.get(akai_midimix="I1", button_mode="toggle")
    if do_record_movie:
        if not movie_recording_started:
            movie_recording_started = 1
            time_stamp = str(time.time())
            os.makedirs('./movies_out', exist_ok=True)
            fp_movie_out = f'./movies_out/movie_{time_stamp}.mp4'
            ms = lt.MovieSaver(fp_movie_out, fps=fps)
        ms.write_frame(image)
    else:
        if movie_recording_started:
            ms.finalize()
        movie_recording_started = 0
    
    # move fract (transition between prompts) forward
    d_fract= meta_input.get(akai_midimix="A1", akai_lpd8="E0", val_min=0.0005, val_max=0.05, val_default=0.05)
    fract += d_fract
    fract = np.clip(fract, 0, 1)
    

        
        
