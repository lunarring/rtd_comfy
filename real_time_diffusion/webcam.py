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
do_compile = False

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

def get_noise_for_modulations(shape):
    return torch.randn(shape, device=pipe.device, generator=torch.Generator(device=pipe.device).manual_seed(1)).half()

modulations = {}
modulations_noise = {}
# for i in range(3):
#     modulations_noise[f'e{i}'] = get_noise_for_modulations(get_sample_shape_unet(f'e{i}'))
#     modulations_noise[f'd{i}'] = get_noise_for_modulations(get_sample_shape_unet(f'd{i}'))
    
# modulations_noise['b0'] = get_noise_for_modulations(get_sample_shape_unet('b0'))
    
prompt_decoder = 'fire'
prompt_embeds_decoder, negative_prompt_embeds_decoder, pooled_prompt_embeds_decoder, negative_pooled_prompt_embeds_decoder = blender.get_prompt_embeds(prompt_decoder, negative_prompt)

last_render_timestamp = time.time()
fract = 0
use_modulated_unet = True


    
#%%
prompt = 'Bizarre creature from Hieronymus Bosch painting "A Garden of Earthly Delights" on a schizophrenic ayahuasca trip'

fract = 0
blender.set_prompt1(prompt)
blender.set_prompt2(prompt)
prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.blend_stored_embeddings(fract)
# prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender
blender.blend_stored_embeddings(fract)
# prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)

# Renderer
renderer = lt.Renderer(width=sz_renderwin[1], height=sz_renderwin[0])

cam_img = cam.get_img()
cam_img = np.flip(cam_img, axis=1)

    
last_frame_tracer = None
    
do_add_noise = True

# fp_movie_out = f'./movies_out/{person_name}.mp4'
# ms = lt.MovieSaver(fp_movie_out, fps=11)



noise_resolution_w = base_w*resolution_factor
noise_resolution_h = base_h*resolution_factor

cam_resolution_w = base_w*8*resolution_factor
cam_resolution_h = base_h*8*resolution_factor

# test resolution
cam_img = cv2.resize(cam_img.astype(np.uint8), (cam_resolution_w, cam_resolution_h))

# fp_aug = 'augs/baloon.png'
# aug_overlay = cv2.imread(fp_aug)[:,:,::-1].copy()
# aug_overlay = cv2.resize(aug_overlay.astype(np.uint8), (cam_resolution_w, cam_resolution_h))

last_cam_img_torch = None

meta_input = lt.MetaInput()
# emodisk = EmoController()

memory_matrix = np.linspace(0.1,0.4,cam_img.shape[1])
memory_matrix = np.expand_dims(np.expand_dims(memory_matrix, 0), -1)
speech_detector = lt.Speech2Text()

# noise
noise_img2img_orig = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda()

torch_last_diffusion_image = torch.from_numpy(cam_img).to('cuda', dtype=torch.float)
image_displacement_accumulated = 0
image_displacement_array_accumulated = None

def get_sample_shape_unet(coord):
    if coord[0] == 'e':
        coef = float(2**int(coord[1]))
        shape = [int(np.ceil(noise_resolution_h/coef)), int(np.ceil(noise_resolution_w/coef))]
    elif coord[0] == 'b':
        shape = [int(np.ceil(noise_resolution_h/4)), int(np.ceil(noise_resolution_w/4))]
    else:
        coef = float(2**(2-int(coord[1])))
        shape = [int(np.ceil(noise_resolution_h/coef)), int(np.ceil(noise_resolution_w/coef))]
        
    return shape


use_pose = False
latents = torch.randn((1,4,noise_resolution_h, noise_resolution_w)).half().cuda()

movie_recording_started = 0

if use_pose:
    client = lt.ZMQPairEndpoint(is_server=False, ip='127.0.0.1', port='5556')

while True:
    do_fix_seed = not meta_input.get(akai_midimix='F3', button_mode='toggle')
    if do_fix_seed:
        torch.manual_seed(0)
        
    noise_img2img_fresh = torch.randn((1,4,noise_resolution_h, noise_resolution_w)).half().cuda()
    
    noise_mixing = meta_input.get(akai_midimix="D0", val_min=0, val_max=1.0, val_default=0)
    noise_img2img = blender.interpolate_spherical(noise_img2img_orig, noise_img2img_fresh, noise_mixing)
    do_cam_coloring = meta_input.get(akai_midimix="G3", button_mode="toggle")
    do_gray_noise = meta_input.get(akai_midimix="G4", button_mode="toggle")
    
    do_record_mic = meta_input.get(keyboard='r', button_mode='pressed_once')
    
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
            stop_recording = False
            fract = 0
            blender.set_prompt1(prompt_prev, negative_prompt)
            blender.set_prompt2(prompt, negative_prompt)
        except Exception as e:
            print(f"fail! {e}")
            
    
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.blend_stored_embeddings(fract)


    
    # # save_midi_settings = meta_input.get(akai_midimix='D4', button_mode='pressed_once')
    # # if save_midi_settings:
        
    #     path_midi_dump = "../submersion/midi_dumps"
    #     fn = None
    #     os.makedirs(path_midi_dump, exist_ok=True)
    #     parameters = []
    #     from datetime import datetime
    #     import yaml
    #     if fn == None:
    #         current_datetime_string = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    #         fn = f'midi_dump_{current_datetime_string}.yml'
    #     fp = os.path.join(path_midi_dump, fn)
    #     for id_, name in meta_input.id_name.items():
    #         value = meta_input.id_value[id_]
    #         parameters.append({'id':id_, 'name':name, 'value':value})
        
    #     parameters.append({'prompt':prompt})
    #     with open(fp, 'w') as file:
    #         yaml.dump(parameters, file)
        # akai_midimix.yaml_dump(path=path_midi_dump, prompt=prompt)
        
    use_image2image = meta_input.get(akai_midimix="I2", button_mode="toggle")
    if use_image2image:
        speed_movie = 1
        cam_img = movie_reader.get_next_frame(speed=int(speed_movie))
        cam_img = cv2.resize(cam_img, (640, 360))
        cam_img = cam_img[:,:,::-1]
        # print(f'using the video {fp_movie}')
    else:
        cam_img = cam.get_img()
        

        
    cam_img = np.flip(cam_img, axis=1)
    
    
    # mask the body
    apply_body_mask = meta_input.get(akai_midimix="E3", button_mode="toggle")
    if apply_body_mask:
        mask_strength = meta_input.get(akai_midimix="E2", val_min=0.0, val_max=1.0, val_default=1)
        mask = np.expand_dims(human_seg.get_mask(cam_img), axis=2)
        cam_img_masked = cam_img * mask
        cam_img = (1-mask_strength)*cam_img + mask_strength * cam_img_masked
        cam_img = cam_img.astype(np.uint8)
    
    # test resolution
    cam_img = cv2.resize(cam_img.astype(np.uint8), (cam_resolution_w, cam_resolution_h))
    
    # do_aug_overlay = meta_input.get(akai_midimix='C3', button_mode='toggle')
    # if do_aug_overlay:
    #     aug_overlay = np.roll(aug_overlay,-10, axis=0)
    #     mask_aug = aug_overlay[:,:,0] != 0
    #     cam_img[mask_aug] = aug_overlay[mask_aug]
    
    strength = meta_input.get(akai_midimix="D2", val_min=0.5, val_max=1.0, val_default=0.5)
    # strength = 0.5
    # num_inference_steps = int(meta_input.get(akai_midimix="C1", val_min=2, val_max=10, val_default=2))
    guidance_scale = 0.5
    guidance_scale = meta_input.get(akai_midimix="C1", val_min=0.001, val_max=1., val_default=0.5)
    # num_inference_steps = meta_input.get(akai_midimix="D1", val_min=2, val_max=6.1, val_default=2)
    # num_inference_steps = int(num_inference_steps)
    num_inference_steps = 2
    
    cam_img_torch = torch.from_numpy(cam_img.copy()).to(latents.device).float()
    
    disable_blur = meta_input.get(akai_midimix="F4", button_mode="toggle")
    if not disable_blur:
        cam_img_torch = blur(cam_img_torch.permute([2,0,1])[None])[0].permute([1,2,0])
    
    if use_pose:
        client_msgs = client.get_messages()
        if client_msgs:
            dict_coco_keypoints = {0:'nose', 1:'left_eye', 2:'right_eye', 3:'left_ear', 4:'right_ear', 5:'left_shoulder', 6:'right_shoulder', 7:'left_elbow', 8:'right_elbow', 9:'left_wrist', 10:'right_wrist', 11:'left_hip', 12:'right_hip', 13:'left_knee', 14:'right_knee', 15:'left_ankle', 16:'right_ankle'}
            dict_keypoint_names = {v: k for k, v in dict_coco_keypoints.items()}
            array_keypoints = client_msgs[0]['keypoints'][0]
            dist_shoulders = array_keypoints[dict_keypoint_names['left_shoulder']][0] - array_keypoints[dict_keypoint_names['right_shoulder']][0]
            dist_wrists = array_keypoints[dict_keypoint_names['left_wrist']][0] - array_keypoints[dict_keypoint_names['right_wrist']][0]
            print("Messages received by client:", client_msgs)
    
    
    # torch_last_diffusion_image = torch.from_numpy(last_diffusion_image).to(cam_img_torch)
    do_zoom = meta_input.get(akai_midimix="H4", akai_lpd8="C0", button_mode="toggle")
    if do_zoom:
        if use_pose:
            try:
                zoom_factor = 1 + ((dist_wrists - dist_shoulders)/dist_shoulders)/10
            except:
                zoom_factor = 1
            print(f'zoom_factor: {zoom_factor}')
        else:
            zoom_factor = meta_input.get(akai_midimix="F0", akai_lpd8="G0", val_min=0.9, val_max=1.1, val_default=1)
        torch_last_diffusion_image = zoom_image_torch(torch_last_diffusion_image, zoom_factor)
    if do_cam_coloring:
        for c in range(3):
            coloring_mask = (torch.rand(cam_img_torch.shape[0], cam_img_torch.shape[1], 1) < 0.8).repeat(1, 1, 3)
            coloring_mask[:,:,c] = 0
            cam_img_torch[coloring_mask] = 255


    if do_add_noise:
        # coef noise
        coef_noise = meta_input.get(akai_midimix="E0", akai_lpd8="E1", val_min=0, val_max=0.5, val_default=0.15)
        # latent_noise_sigma = meta_input.get(akai_midimix="E1", akai_lpd8="E2", val_min=0.7, val_max=1.3, val_default=1)
        
        if not do_gray_noise:
            
            do_gaussian_noise = meta_input.get(akai_midimix="E4", button_mode="toggle")
            if do_gaussian_noise:
                t_rand = (torch.randn(cam_img_torch.shape[0], cam_img_torch.shape[1], 3, device=cam_img_torch.device) - 0.5) * coef_noise * 255
            else:
                t_rand = (torch.rand(cam_img_torch.shape[0], cam_img_torch.shape[1], 3, device=cam_img_torch.device) - 0.5) * coef_noise * 255
                
            # t_rand[t_rand < 0.5] = 0


        else:
            t_rand = (torch.rand(cam_img_torch.shape, device=cam_img_torch.device)[:,:,0].unsqueeze(2) - 0.5) * coef_noise * 255
        cam_img_torch += t_rand
        torch_last_diffusion_image += t_rand
        # cam_img_torch += (torch.rand(cam_img_torch.shape, device=cam_img_torch.device)[:,:,0].unsqueeze(2) - 0.5) * coef_noise * 255 * 5
    

    do_accumulate_acid = meta_input.get(akai_midimix="C4", akai_lpd8="B0", button_mode="toggle")
    do_local_accumulate_acid = meta_input.get(akai_midimix="D4", button_mode="toggle")
    invert_accumulate_acid = meta_input.get(akai_midimix="D3", akai_lpd8="B1", button_mode="toggle")
    # acid_persistence = meta_input.get(akai_midimix="D1", val_min=0.01, val_max=0.99, val_default=0.5)
    # acid_decay = meta_input.get(akai_midimix="D2", val_min=0.01, val_max=0.5, val_default=0.2)
    
    if do_accumulate_acid:
        ## displacement controlled acid
        if last_cam_img_torch is None:
            last_cam_img_torch = cam_img_torch
        acid_gain = meta_input.get(akai_midimix="C0", akai_lpd8="F0", val_min=0, val_max=0.5, val_default=0.05)
            
        image_displacement_array = ((cam_img_torch - last_cam_img_torch)/255)**2
        
        if do_local_accumulate_acid:
            image_displacement_array = (1-image_displacement_array*100)
            image_displacement_array = image_displacement_array.clamp(0)
            if image_displacement_array_accumulated == None:
                image_displacement_array_accumulated = torch.zeros_like(image_displacement_array)           
            image_displacement_array_accumulated[image_displacement_array>=0.5] += 2e-2
            image_displacement_array_accumulated[image_displacement_array<0.5] -= 2e-1
            image_displacement_array_accumulated = image_displacement_array_accumulated.clamp(0)
            
            image_displacement_array_accumulated = image_displacement_array_accumulated.mean(2, keepdims=True)
            image_displacement_array_accumulated = image_displacement_array_accumulated.repeat([1,1,3])
            
            image_displacement_array_accumulated -= image_displacement_array_accumulated.min()
            image_displacement_array_accumulated /= image_displacement_array_accumulated.max()
            
            if invert_accumulate_acid:
                acid_array = 1-image_displacement_array_accumulated
                acid_array[acid_array<0.05]=0.05
                acid_array *= acid_gain                
            else:
                acid_array = (image_displacement_array_accumulated)*acid_gain

        
        else:
            image_displacement = image_displacement_array.mean()
            image_displacement = (1-image_displacement*100)
            if image_displacement < 0:
                image_displacement = 0
                
            if image_displacement >= 0.5:
                image_displacement_accumulated += 2e-2
            else:
                image_displacement_accumulated -= 4e-1

            if image_displacement_accumulated < 0:
                image_displacement_accumulated = 0
            
            if invert_accumulate_acid:
                acid_strength = max(0.1, 1 - image_displacement_accumulated)
            else:
                acid_strength = 0.1 + image_displacement_accumulated * 1
            acid_strength *= acid_gain
            acid_strength = min(1, acid_strength)
        last_cam_img_torch = cam_img_torch.clone()
    else:
        acid_strength = meta_input.get(akai_midimix="C0", akai_lpd8="F0", val_min=0, val_max=0.8, val_default=0.11)
        
    acid_freq = meta_input.get(akai_midimix="F2", val_min=0, val_max=10.0, val_default=0)
    # if acid_freq > 0:
    #     acid_strength = (np.sin(acid_freq*float(time.time())) + 1)/2
        
    
    # just a test
    # cam_img_torch = (1.-acid_strength)*cam_img_torch + acid_strength*torch.from_numpy(last_diffusion_image).to(cam_img_torch)
    if do_accumulate_acid and do_local_accumulate_acid:
        cam_img_torch = (1.-acid_array)*cam_img_torch + acid_array*torch_last_diffusion_image
    else:
        cam_img_torch = (1.-acid_strength)*cam_img_torch + acid_strength*torch_last_diffusion_image
    # if meta_input.get(akai_midimix='E4', button_mode='pressed_once'):
    #     xxx
    cam_img_torch = torch.clamp(cam_img_torch, 0, 255)
    # paint_decay = 0.999
    # color_strenght = 1
    # if apply_body_mask:
    #     mask_torch = F.interpolate(torch.from_numpy(mask).to(latents.device).float().permute(2,0,1).unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False).squeeze(axis=0).permute(1,2,0)
    #     if canvas.shape != cam_img_torch.shape:
    #         canvas = torch.zeros_like(cam_img_torch)
        

    color_matching = meta_input.get(akai_lpd8="G1", akai_midimix="C2", val_min=0, val_max=1, val_default=0.0)
    if color_matching > 0.01:
        if apply_body_mask:
            mask_torch = F.interpolate(torch.from_numpy(mask).to(latents.device).float().permute(2,0,1).unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False).squeeze(axis=0).permute(1,2,0)
            torch_last_diffusion_image_masked = torch_last_diffusion_image*mask_torch
            torch_last_diffusion_image = (1-mask_strength)*torch_last_diffusion_image + mask_strength * torch_last_diffusion_image_masked
            cam_img_torch, _ = multi_match_gpu([cam_img_torch, torch_last_diffusion_image], weights=[1-color_matching, color_matching], clip_max='auto', gpu=0,  is_input_tensor=True)
        else:
            cam_img_torch, _ = multi_match_gpu([cam_img_torch, torch_last_diffusion_image], weights=[1-color_matching, color_matching], clip_max='auto', gpu=0,  is_input_tensor=True)

    cam_img = cam_img_torch.cpu().numpy()
    
    # # mask the body
    # apply_body_mask = meta_input.get(akai_midimix="E3", button_mode="toggle")
    # if apply_body_mask:
    #     mask = human_seg.get_mask(cam_img)
    #     cam_img *= np.expand_dims(mask, axis=2)    
    
    if use_modulated_unet:
        mod_samp = meta_input.get(akai_midimix="H2", val_min=0, val_max=10, val_default=0)
        modulations['b0_samp'] = torch.tensor(mod_samp, device=latents.device)
        modulations['e2_samp'] = torch.tensor(mod_samp, device=latents.device)
        
        mod_emb = meta_input.get(akai_midimix="H1", akai_lpd8="F1", val_min=1, val_max=10, val_default=2)
        modulations['b0_emb'] = torch.tensor(mod_emb, device=latents.device)
        modulations['e2_emb'] = torch.tensor(mod_emb, device=latents.device)
        
        fract_mod = meta_input.get(akai_midimix="G0", val_default=0, val_max=2, val_min=0)
        if fract_mod > 1:
            modulations['d*_extra_embeds'] = prompt_embeds_decoder    
        else:
            modulations['d*_extra_embeds'] = prompt_embeds
            
        modulations['modulations_noise'] = modulations_noise
        
    if use_modulated_unet:
        cross_attention_kwargs ={}
        cross_attention_kwargs['modulations'] = modulations
        # cross_attention_kwargs['latent_noise_sigma'] = latent_noise_sigma
    else:
        cross_attention_kwargs = None
    
    use_debug_overlay = meta_input.get(akai_midimix="H3", akai_lpd8="D1", button_mode="toggle")
    if use_debug_overlay:
        image = cam_img.astype(np.uint8)
        if do_local_accumulate_acid:
            try:
                image = (image_displacement_array_accumulated*255).cpu().numpy().astype(np.uint8)
            except Exception as e:
                print(f"bad error! {e}")
    else:
        image = pipe(image=Image.fromarray(cam_img.astype(np.uint8)), 
                      latents=latents, num_inference_steps=num_inference_steps, strength=strength, 
                      guidance_scale=guidance_scale, prompt_embeds=prompt_embeds, 
                      negative_prompt_embeds=negative_prompt_embeds, 
                      pooled_prompt_embeds=pooled_prompt_embeds, 
                      negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, noise_img2img=noise_img2img).images[0]
        
    time_difference = time.time() - last_render_timestamp
    last_render_timestamp = time.time()
    
    fps = np.round(1/time_difference)
    lt.dynamic_print(f'fps: {fps}')
    try:
        torch_last_diffusion_image = torchvision.transforms.functional.pil_to_tensor(image).to(latents.device, dtype=torch.float).permute(1,2,0)
    except:
        torch_last_diffusion_image = torch.from_numpy(image).to(latents.device, dtype=torch.float)
    # last_diffusion_image = np.array(image, dtype=np.float32)
    
    do_antishift = meta_input.get(akai_midimix="A4", akai_lpd8="D0", button_mode="toggle")
    x_shift = int(meta_input.get(akai_midimix="B0", akai_lpd8="H0", val_default=0, val_max=10, val_min=-10))
    y_shift = int(meta_input.get(akai_midimix="B1", akai_lpd8="H1", val_default=0, val_max=10, val_min=-10))
    if do_antishift:
        torch_last_diffusion_image = torch.roll(torch_last_diffusion_image, (y_shift, x_shift), (0,1))
        

    
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
    
    # move fract forward
    d_fract_embed = meta_input.get(akai_midimix="A1", akai_lpd8="E0", val_min=0.0005, val_max=0.05, val_default=0.05)
    fract += d_fract_embed
    fract = np.clip(fract, 0, 1)
    


        
        
