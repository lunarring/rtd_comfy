#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers import AutoencoderTiny
import datetime
import time
import lunar_tools as lt
from PIL import Image
import numpy as np
from prompt_blender import PromptBlender
from human_seg import HumanSeg
import torchvision
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False 


def get_diffusion_dimensions(height_diffusion_desired, width_diffusion_desired, latent_div=16, autoenc_div=8):
    """
    This function calculates the correct dimensions for the diffusion process based on the desired dimensions. 
    It ensures that the dimensions are divisible by the latent_div and autoenc_div parameters. 
    If the desired dimensions are not divisible, it adjusts them to the nearest divisible number and returns the corrected dimensions.

    Args:
        height_diffusion_desired (int): The desired height for the diffusion process.
        width_diffusion_desired (int): The desired width for the diffusion process.
        latent_div (int, optional): The divisor for the latent dimensions. Defaults to 16.
        autoenc_div (int, optional): The divisor for the autoencoder dimensions. Defaults to 8.

    Returns:
        height_diffusion_corrected (int): The corrected height for the diffusion process.
        width_diffusion_corrected (int): The corrected width for the diffusion process.
        height_latents (int): The height of the latent space.
        width_latents (int): The width of the latent space.
    """
    height_latents = round(height_diffusion_desired / autoenc_div)
    height_latents = round(latent_div * height_latents / latent_div)
    height_diffusion_corrected = int(height_latents * autoenc_div)

    width_latents = round(width_diffusion_desired / autoenc_div)
    width_latents = round(latent_div * width_latents / latent_div)
    width_diffusion_corrected = int(width_latents * autoenc_div)

    if height_diffusion_corrected != height_diffusion_desired or width_diffusion_corrected != width_diffusion_desired:
        print(f"Autocorrected the given dimensions. Corrected: ({height_diffusion_corrected}, {width_diffusion_corrected}), Given: ({height_diffusion_desired}, {width_diffusion_desired})")

    return height_diffusion_corrected, width_diffusion_corrected, height_latents, width_latents
        



#%% VARS
shape_cam = (300,400)
height_diffusion_desired = 768
width_diffusion_desired = 1024
do_compile = False

sz_renderwin = (int(512*2.09), int(512*3.85))

guidance_scale = 0.5
strength = 0.5
num_inference_steps = 2

resizing_factor_humanseg = 0.5

negative_prompt = "blurry, bland, black and white, monochromatic"
model_turbo = "stabilityai/sdxl-turbo"
model_vae = "madebyollin/taesdxl"

#%% Inits
height_diffusion, width_diffusion, height_latents, width_latents = get_diffusion_dimensions(height_diffusion_desired, width_diffusion_desired)
cam = lt.WebCam(cam_id=-1, shape_hw=shape_cam)
cam.autofocus_enable()

# Diffusion Pipe
pipe = AutoPipelineForImage2Image.from_pretrained(model_turbo, torch_dtype=torch.float16, variant="fp16", local_files_only=False)
pipe.to("cuda")
pipe.vae = AutoencoderTiny.from_pretrained(model_vae, torch_device='cuda', torch_dtype=torch.float16, local_files_only=False)
pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)

if do_compile:
    from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
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
human_seg = HumanSeg(resizing_factor=resizing_factor_humanseg)

# Promptblender
blender = PromptBlender(pipe)

# Controls (keyboard, midi devices)
meta_input = lt.MetaInput()

# Speech detector
speech_detector = lt.Speech2Text()

# Render windows
renderer = lt.Renderer(width=sz_renderwin[1], height=sz_renderwin[0])

# Median blurring
blur = lt.MedianBlur((7, 7))


# runtime vars
prompt = 'Bizarre creature from Hieronymus Bosch painting "A Garden of Earthly Delights" on a schizophrenic ayahuasca trip'
fract = 0
last_render_timestamp = time.time()

blender.set_prompt1(prompt)
blender.set_prompt2(prompt)
prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.blend_stored_embeddings(fract)

cam_img = cam.get_img()
cam_img = np.flip(cam_img, axis=1)
cam_img = lt.resize(cam_img, size=(height_diffusion, width_diffusion))

# noise
noise_img2img_orig = torch.randn((1, 4, height_latents, width_latents)).half().cuda()
torch_last_diffusion_image = torch.from_numpy(cam_img).to('cuda', dtype=torch.float)
latents = torch.randn((1,4,height_latents, width_latents)).half().cuda()

movie_recording_started = False

while True:
    do_fix_seed = not meta_input.get(akai_midimix='F3', button_mode='toggle')
    if do_fix_seed:
        torch.manual_seed(420)
    
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
            
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.blend_stored_embeddings(fract)
    
    cam_img = cam.get_img()
    cam_img = np.flip(cam_img, axis=1)
    
    # mask the body
    apply_body_mask = meta_input.get(akai_midimix="E3", akai_lpd8="A0", button_mode="toggle", val_default=True)
    if apply_body_mask:
        mask_strength = meta_input.get(akai_midimix="E2", val_min=0.0, val_max=1.0, val_default=1)
        human_seg.get_mask(cam_img)
        cam_img = human_seg.apply_mask(cam_img, mask_strength=mask_strength)
    
    # median filter
    cam_img = lt.resize(cam_img, size=(height_diffusion, width_diffusion))
    cam_img_torch = torch.from_numpy(cam_img.copy()).to(latents.device).float()
    cam_img_torch = blur(cam_img_torch.permute([2,0,1])[None])[0].permute([1,2,0])


    # add noise
    coef_noise = meta_input.get(akai_lpd8="E1", val_min=0, val_max=0.5, val_default=0.15)
    t_rand = (torch.rand(cam_img_torch.shape[0], cam_img_torch.shape[1], 3, device=cam_img_torch.device) - 0.5) * coef_noise * 255
    cam_img_torch += t_rand
    torch_last_diffusion_image += t_rand

    # acid
    acid_strength = meta_input.get(akai_lpd8="F0", val_min=0, val_max=0.8, val_default=0.11)
    cam_img_torch = (1.-acid_strength)*cam_img_torch + acid_strength*torch_last_diffusion_image
    cam_img_torch = torch.clamp(cam_img_torch, 0, 255)
    cam_img = cam_img_torch.cpu().numpy()
    
    use_debug_overlay = meta_input.get(akai_midimix="H3", akai_lpd8="D1", button_mode="toggle")
    if use_debug_overlay:
        render_image = cam_img.astype(np.uint8)

    else:
        render_image = pipe(image=Image.fromarray(cam_img.astype(np.uint8)), 
                      latents=latents, num_inference_steps=num_inference_steps, strength=strength, 
                      guidance_scale=guidance_scale, prompt_embeds=prompt_embeds, 
                      negative_prompt_embeds=negative_prompt_embeds, 
                      pooled_prompt_embeds=pooled_prompt_embeds, 
                      negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, noise_img2img=noise_img2img_orig).images[0]
        
    time_difference = time.time() - last_render_timestamp
    last_render_timestamp = time.time()
    fps = 1/time_difference
    lt.dynamic_print(f'fps: {fps:2.2f}')

    torch_last_diffusion_image = torchvision.transforms.functional.pil_to_tensor(render_image).to(latents.device, dtype=torch.float).permute(1,2,0)
    
    # Render the image
    renderer.render(render_image)

    do_record_movie = meta_input.get(akai_midimix="I1", akai_lpd8="D0", button_mode="toggle")
    if do_record_movie:
        if not movie_recording_started:
            movie_recording_started = True
            time_stamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
            os.makedirs('./movies_out', exist_ok=True)
            fp_movie_out = f'./recordings/recording_{time_stamp}.mp4'
            ms = lt.MovieSaverThreaded(fp_movie_out, fps=fps)
        ms.write_frame(render_image)
    else:
        if movie_recording_started:
            ms.finalize()
        movie_recording_started = False
    
    # move fract forward for smooth prompt transitions
    d_fract_embed = meta_input.get(akai_midimix="A1", akai_lpd8="E0", val_min=0.0005, val_max=0.05, val_default=0.05)
    fract += d_fract_embed
    fract = np.clip(fract, 0, 1)
    


        
        
