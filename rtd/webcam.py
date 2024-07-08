#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import torch
import datetime
import time
import lunar_tools as lt
from PIL import Image
import numpy as np
from human_seg import HumanSeg
from diffusion_engine import DiffusionEngine
from embeddings_mixer import EmbeddingsMixer
import torchvision
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False 

#%% VARS
shape_cam = (300,400)
height_diffusion_desired = 768
width_diffusion_desired = 1024
do_compile = True

sz_renderwin = (int(height_diffusion_desired*2), int(width_diffusion_desired*2))

guidance_scale = 0.5
strength = 0.5
num_inference_steps = 2
resizing_factor_humanseg = 0.5

negative_prompt = "blurry, bland, black and white, monochromatic"

#%% Inits
# Diffusion Engine
de = DiffusionEngine(do_compile=do_compile, height_diffusion_desired=height_diffusion_desired, width_diffusion_desired=width_diffusion_desired)
de.set_num_inference_steps(num_inference_steps)
de.set_guidance_scale(guidance_scale)
de.set_strength(strength)

# Webcam Input
cam = lt.WebCam(cam_id=-1, shape_hw=shape_cam)
cam.autofocus_enable()

# human body segmentation
human_seg = HumanSeg(resizing_factor=resizing_factor_humanseg)

# Embeddings Mixer
em = EmbeddingsMixer(de.pipe)

# Controls (keyboard, midi devices)
meta_input = lt.MetaInput()

# Speech detector
speech_detector = lt.Speech2Text()

# Render windows
renderer = lt.Renderer(width=sz_renderwin[1], height=sz_renderwin[0])

# Median blurring
blur = lt.MedianBlur((7, 7))

# initialize embeddings
prompt = 'Bizarre creature from Hieronymus Bosch painting "A Garden of Earthly Delights" on a schizophrenic ayahuasca trip'
fract = 0
embeds1 = em.encode_prompt(prompt)
embeds2 = embeds1
de.set_embeddings(embeds1)

# runtime vars
cam_img = cam.get_img()
cam_img = np.flip(cam_img, axis=1)
cam_img = lt.resize(cam_img, size=(de.height_diffusion, de.width_diffusion))

last_render_timestamp = time.time()
torch_last_diffusion_image = torch.from_numpy(cam_img).to('cuda', dtype=torch.float)

movie_recording_started = False

while True:
    torch.manual_seed(420)
    
    # Update the prompts with button
    do_record_mic = meta_input.get(keyboard='r', akai_lpd8="A1", button_mode='held_down')
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
                embeds1 = em.encode_prompt(prompt_prev, negative_prompt)
                embeds2 = em.encode_prompt(prompt, negative_prompt)
                
            except Exception as e:
                print(f"FAIL {e}")
    # Mix prompts
    embeds = em.blend_two_embeds(embeds1, embeds2, fract)
    de.set_embeddings(embeds)
    
    # Get cam image
    cam_img = cam.get_img()
    cam_img = np.flip(cam_img, axis=1)
    
    # mask the body
    apply_body_mask = meta_input.get(akai_midimix="E3", akai_lpd8="A0", button_mode="toggle", val_default=True)
    if apply_body_mask:
        mask_strength = meta_input.get(akai_midimix="E2", val_min=0.0, val_max=1.0, val_default=1)
        human_seg.get_mask(cam_img)
        cam_img = human_seg.apply_mask(cam_img, mask_strength=mask_strength)
    
    # median filter
    cam_img = lt.resize(cam_img, size=(de.height_diffusion, de.width_diffusion))
    cam_img_torch = torch.from_numpy(cam_img.copy()).to(de.device).float()
    cam_img_torch = blur(cam_img_torch.permute([2,0,1])[None])[0].permute([1,2,0])

    # add noise
    coef_noise = meta_input.get(akai_lpd8="E1", val_min=0, val_max=0.3, val_default=0.15)
    t_rand = (torch.rand(cam_img_torch.shape[0], cam_img_torch.shape[1], 3, device=cam_img_torch.device) - 0.5) * coef_noise * 255
    cam_img_torch += t_rand
    torch_last_diffusion_image += t_rand

    # acid
    acid_strength = meta_input.get(akai_lpd8="F0", val_min=0, val_max=0.8, val_default=0.1)
    cam_img_torch = (1.-acid_strength)*cam_img_torch + acid_strength*torch_last_diffusion_image
    cam_img_torch = torch.clamp(cam_img_torch, 0, 255)
    cam_img = cam_img_torch.cpu().numpy()

    de.set_input_image(cam_img)
    
    use_debug_overlay = meta_input.get(akai_midimix="H3", akai_lpd8="D1", button_mode="toggle")
    if use_debug_overlay:
        render_image = cam_img.astype(np.uint8)
    else:
        render_image = de.generate()
        torch_last_diffusion_image = torchvision.transforms.functional.pil_to_tensor(render_image).to(de.device, dtype=torch.float).permute(1,2,0)
     
    # fps display
    time_difference = time.time() - last_render_timestamp
    last_render_timestamp = time.time()
    fps = 1/time_difference
    lt.dynamic_print(f'fps: {fps:2.2f}')

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
    


        
        
