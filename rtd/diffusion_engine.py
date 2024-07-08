import numpy as np
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.models import UNet2DConditionModel
from diffusers import AutoencoderTiny
import torch
from PIL import Image
import lunar_tools as lt
from torch.nn import functional as F
import torch
import torch.nn as nn
import torch.utils.checkpoint
from embeddings_mixer import EmbeddingsMixer
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    GLIGENTextBoundingboxProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import (
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    get_down_block,
    get_up_block,
)

def torch_resample(tex, grid,padding_mode="reflection", mode='bilinear'):
    if len(tex.shape) == 3:                        # add singleton to batch dim
        return F.grid_sample(tex.view((1,)+tex.shape),grid.view((1,)+grid.shape),padding_mode=padding_mode,mode=mode)[0,:,:,:].permute([1,2,0])
    elif len(tex.shape) == 4:
        return F.grid_sample(tex,grid.view((1,)+grid.shape),padding_mode=padding_mode,mode=mode)[0,:,:,:].permute([1,2,0])
    else:
        raise ValueError('torch_resample: bad input dims')
        
@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None        

def forward_modulated(
    self,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    modulations = {},
) -> Union[UNet2DConditionOutput, Tuple]:
    r"""
    The [`UNet2DConditionModel`] forward method.

    Args:
        sample (`torch.FloatTensor`):
            The noisy input tensor with the following shape `(batch, channel, height, width)`.
        timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
        encoder_hidden_states (`torch.FloatTensor`):
            The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
        class_labels (`torch.Tensor`, *optional*, defaults to `None`):
            Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
        timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
            Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
            through the `self.time_embedding` layer to obtain the timestep embeddings.
        attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
            is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
            negative values to the attention scores corresponding to "discard" tokens.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        added_cond_kwargs: (`dict`, *optional*):
            A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
            are passed along to the UNet blocks.
        down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
            A tuple of tensors that if specified are added to the residuals of down unet blocks.
        mid_block_additional_residual: (`torch.Tensor`, *optional*):
            A tensor that if specified is added to the residual of the middle unet block.
        encoder_attention_mask (`torch.Tensor`):
            A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
            `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
            which adds large negative values to the attention scores corresponding to "discard" tokens.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
            tuple.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].
        added_cond_kwargs: (`dict`, *optional*):
            A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
            are passed along to the UNet blocks.
        down_block_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
            additional residuals to be added to UNet long skip connections from down blocks to up blocks for
            example from ControlNet side model(s)
        mid_block_additional_residual (`torch.Tensor`, *optional*):
            additional residual to be added to UNet mid block output, for example from ControlNet side model
        down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
            additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)

    Returns:
        [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            If `return_dict` is True, an [`~models.unet_2d_condition.UNet2DConditionOutput`] is returned, otherwise
            a `tuple` is returned where the first element is the sample tensor.
    """
    
    # By default samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            # Forward upsample size to force interpolation output size.
            forward_upsample_size = True
            break

    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])

    t_emb = self.time_proj(timesteps)

    # `Timesteps` does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=sample.dtype)

    emb = self.time_embedding(t_emb, timestep_cond)
    aug_emb = None

    if self.class_embedding is not None:
        if class_labels is None:
            raise ValueError("class_labels should be provided when num_class_embeds > 0")

        if self.config.class_embed_type == "timestep":
            class_labels = self.time_proj(class_labels)

            # `Timesteps` does not contain any weights and will always return f32 tensors
            # there might be better ways to encapsulate this.
            class_labels = class_labels.to(dtype=sample.dtype)

        class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    if self.config.addition_embed_type == "text":
        aug_emb = self.add_embedding(encoder_hidden_states)
    elif self.config.addition_embed_type == "text_image":
        # Kandinsky 2.1 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
            )

        image_embs = added_cond_kwargs.get("image_embeds")
        text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
        aug_emb = self.add_embedding(text_embs, image_embs)
    elif self.config.addition_embed_type == "text_time":
        # SDXL - style
        if "text_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
            )
        text_embeds = added_cond_kwargs.get("text_embeds")
        if "time_ids" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
            )
        time_ids = added_cond_kwargs.get("time_ids")
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)
    elif self.config.addition_embed_type == "image":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
            )
        image_embs = added_cond_kwargs.get("image_embeds")
        aug_emb = self.add_embedding(image_embs)
    elif self.config.addition_embed_type == "image_hint":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
            )
        image_embs = added_cond_kwargs.get("image_embeds")
        hint = added_cond_kwargs.get("hint")
        aug_emb, hint = self.add_embedding(image_embs, hint)
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
    elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
        # Kadinsky 2.1 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
            )

        image_embeds = added_cond_kwargs.get("image_embeds")
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
    elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
            )
        image_embeds = added_cond_kwargs.get("image_embeds")
        encoder_hidden_states = self.encoder_hid_proj(image_embeds)
    elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
            )
        image_embeds = added_cond_kwargs.get("image_embeds")
        image_embeds = self.encoder_hid_proj(image_embeds).to(encoder_hidden_states.dtype)
        encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds], dim=1)

    # 2. pre-process
    sample = self.conv_in(sample)

    # 2.5 GLIGEN position net
    if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

    # 3. down
    lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)

    is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
    # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
    is_adapter = down_intrablock_additional_residuals is not None
    # maintain backward compatibility for legacy usage, where
    #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
    #       but can only use one or the other
    if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
        deprecate(
            "T2I should not use down_block_additional_residuals",
            "1.3.0",
            "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                   and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                   for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
            standard_warn=False,
        )
        down_intrablock_additional_residuals = down_block_additional_residuals
        is_adapter = True
        
    if cross_attention_kwargs is not None and 'modulations' in cross_attention_kwargs:
        modulations = cross_attention_kwargs['modulations']
        cross_attention_kwargs = None
        
        
    # modulations
    if modulations is None:
        modulations = {}
    
    down_block_res_samples = (sample,)
    for i, downsample_block in enumerate(self.down_blocks):
        
        if f'e{i}_samp' in modulations:
            noise_coef = modulations[f'e{i}_samp']
            noise = modulations['modulations_noise'][f'e{i}']
            sample += noise * noise_coef
        
        if f'e{i}_emb' in modulations:
            encoder_state_mod = modulations[f'e{i}_emb']
        else:
            encoder_state_mod = 1

        if f'e{i}_acid' in modulations:
            amp_field, warp_field = modulations[f'e{i}_acid']
            sample *= (1+amp_field)            
        
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb*encoder_state_mod,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                sample += down_intrablock_additional_residuals.pop(0)

        down_block_res_samples += res_samples
        
    if is_controlnet:
        new_down_block_res_samples = ()

        for down_block_res_sample, down_block_additional_residual in zip(
            down_block_res_samples, down_block_additional_residuals
        ):
            down_block_res_sample = down_block_res_sample + down_block_additional_residual
            new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = new_down_block_res_samples
        

    # 4. mid
    if self.mid_block is not None:
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            if 'b0_samp' in modulations:
                noise_coef = modulations['b0_samp']
                noise = modulations['modulations_noise']['b0']
                sample += noise * noise_coef
            
            if 'b0_emb' in modulations:
                encoder_state_mod = modulations['b0_emb']
            else:
                encoder_state_mod = 1
                
            if 'b0_acid' in modulations:
                amp_field, warp_field = modulations['b0_acid']
                
                # sample *= (1+amp_field*10)
                sample = torch_resample(sample.float(), ((warp_field * 2) - 1)).permute([2,0,1])[None].half()
                
            sample = self.mid_block(
                sample,
                emb*encoder_state_mod,
                encoder_hidden_states=encoder_hidden_states,
                #encoder_hidden_states = modulations['d*_extra_embeds'],
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = self.mid_block(sample, emb)
            
        # To support T2I-Adapter-XL
        if (
            is_adapter
            and len(down_intrablock_additional_residuals) > 0
            and sample.shape == down_intrablock_additional_residuals[0].shape
        ):
            sample += down_intrablock_additional_residuals.pop(0)
            
    if is_controlnet:
        sample = sample + mid_block_additional_residual

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
        is_final_block = i == len(self.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]
            
        if f'd{i}_samp' in modulations:
            noise_coef = modulations[f'd{i}_samp']
            noise = modulations['modulations_noise'][f'd{i}']
            sample += noise * noise_coef
        
        if f'd{i}_emb' in modulations:
            encoder_state_mod = modulations[f'd{i}_emb']
        else:
            encoder_state_mod = 1
            
        if f'd{i}_acid' in modulations:
            amp_field, warp_field = modulations[f'd{i}_acid']
            sample *= (1+amp_field)
            
        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            if f'd{i}_extra_embeds' in modulations:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb*encoder_state_mod,
                    res_hidden_states_tuple=res_samples,
                    #encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states = modulations[f'd{i}_extra_embeds'],
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb*encoder_state_mod,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )                
            
            
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
                scale=lora_scale,
            )

        # if i == 0:
        #     amp = 1e-2
        #     resample_grid = acidman.do_acid(sample[0].float().permute([1,2,0]), amp)
        #     amp_mod = (resample_grid - acidman.identity_resample_grid)     
            
        #     sample = torch_resample(sample.float(), ((resample_grid * 2) - 1)).permute([2,0,1])[None].half()
        #     # sample *= (1+amp_mod[:,:,0][None][None])
        #     # sample += amp_mod[:,:,0][None][None]


        # if i == 1:
        #     if par_container.sample is None:
        #         par_container.sample = sample        
                
        #     if use_prev_emb:
        #         ramp = torch.linspace(0,1,sample.shape[2], device=sample.device).half()
        #         ramp = ramp[None][None][None]
        #         sample = ramp * sample + (1 - ramp)*par_container.sample
            
    # 6. post-process
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (sample,)

    return UNet2DConditionOutput(sample=sample)

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
        print(f"Autocorrected the desired dimensions. Corrected: ({height_diffusion_corrected}, {width_diffusion_corrected}), Desired: ({height_diffusion_desired}, {width_diffusion_desired})")

    return height_diffusion_corrected, width_diffusion_corrected, height_latents, width_latents


class DiffusionEngine():
    """
    The DiffusionEngine class is used to initialize the diffusion process.
    It provides methods to set the desired dimensions for the diffusion process and to compile the model pipeline if needed.

    Attributes:
        height_diffusion_desired (int): The desired height for the diffusion process in pixel space. Defaults to 512. 
                                        It will be corrected to ensure compatibility with the diffusion engine.
        width_diffusion_desired (int): The desired width for the diffusion process in pixel space. Defaults to 512. 
                                        It will be corrected to ensure compatibility with the diffusion engine.
        use_image2image (bool): A flag to determine whether to use the image-to-image pipeline or text-to-image.
        do_compile (bool): A flag to determine whether to compile the model pipeline. Defaults to False.
    """
    def __init__(
        self,
        height_diffusion_desired = 512,
        width_diffusion_desired = 512,
        use_image2image = True,
        use_tinyautoenc = True,
        device = 'cuda',
        hf_model = 'stabilityai/sdxl-turbo',
        do_compile = False, 
    ):
        height_diffusion_corrected, width_diffusion_corrected, height_latents, width_latents = get_diffusion_dimensions(height_diffusion_desired, width_diffusion_desired)
        self.height_latents = height_latents
        self.width_latents = width_latents
        self.height_diffusion = height_diffusion_corrected
        self.width_diffusion = width_diffusion_corrected
        self.do_compile = do_compile
        self.use_tinyautoenc = use_tinyautoenc
        self.device = device
        self.hf_model = hf_model

        self.num_inference_steps = None
        self.guidance_scale = 0.0
        self.strength = 0.5
        self.latents = None
        self.embeds = None
        self.image_init = None

        self.use_image2image = use_image2image
        if self.use_image2image:
            self._init_image2image()
        else:
            self._init_text2image()


    def _init_image2image(self):
        """
        This method initializes the image-to-image pipeline. It loads the pretrained model from huggingface hf_model
        with a torch_dtype of float16 and variant "fp16". The model is loaded from local files only. 
        The number of inference steps is set to 2 and the pipeline is initialized with the loaded model.
        """
        try:
            pipe = AutoPipelineForImage2Image.from_pretrained(self.hf_model, torch_dtype=torch.float16, variant="fp16", local_files_only=True)
        except Exception as e:
            pipe = AutoPipelineForImage2Image.from_pretrained(self.hf_model, torch_dtype=torch.float16, variant="fp16", local_files_only=False)
        self.num_inference_steps = 2
        self._init_pipe(pipe)

    def _init_text2image(self):
        """
        This method initializes the text-to-image pipeline. It loads the pretrained model from huggingface hf_model
        with a torch_dtype of float16 and variant "fp16". The model is loaded from local files only. 
        The number of inference steps is set to 1 and the pipeline is initialized with the loaded model.
        """
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(self.hf_model, torch_dtype=torch.float16, variant="fp16", local_files_only=True)
        except Exception as e:
            pipe = AutoPipelineForText2Image.from_pretrained(self.hf_model, torch_dtype=torch.float16, variant="fp16", local_files_only=False)
        self.num_inference_steps = 1
        self._init_pipe(pipe)

    def _init_pipe(self, pipe):
        """
        This method initializes the pipeline with the given pipe. It sets the device for the pipe, 
        initializes the autoencoder, sets the forward method for the unet, and compiles the pipe if necessary.
        Finally, it sets the latents for the pipeline.
        """

        pipe.to(self.device)
        if self.use_tinyautoenc:
            pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device=self.device, torch_dtype=torch.float16)
        pipe.vae = pipe.vae.cuda()
        pipe.set_progress_bar_config(disable=True)
        pipe.unet.forward = forward_modulated.__get__(pipe.unet, UNet2DConditionModel)
            
        if self.do_compile:
            from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)    
            pipe.enable_xformers_memory_efficient_attention()
            config = CompilationConfig.Default()
            config.enable_xformers = True
            config.enable_triton = True
            config.enable_cuda_graph = True
            # config.enable_jit = True
            # config.enable_jit_freeze = True
            # config.trace_scheduler = True
            # config.enable_cnn_optimization = True
            # config.preserve_parameters = False
            # config.prefer_lowp_gemm = True
            pipe = compile(pipe, config)
        self.pipe = pipe
        self.set_latents()


    def set_latents(self, latents=None):
        """
        This method sets the latents for the pipeline. If no latents are provided, it generates new latents.
        """
        if latents is None:
            latents = self.get_latents()
        self.latents = latents

    def get_latents(self):
        """Generates and returns latents."""
        return torch.randn((1, 4, self.height_latents, self.width_latents)).half().cuda()

    def set_num_inference_steps(self, num_inference_steps):
        """Sets the number of inference steps."""
        self.num_inference_steps = int(num_inference_steps)

    def set_guidance_scale(self, guidance_scale):
        """Sets the guidance scale."""
        self.guidance_scale = float(guidance_scale)

    def set_strength(self, strength):
        """Sets the strength, ensuring it's greater than 1/num_inference_steps."""
        # assert strength > 1/self.num_inference_steps, "Increase strength!"
        self.strength = float(strength)

    def set_input_image(self, image_init):
        """Sets the input image, resizing if necessary."""
        if not isinstance(image_init, Image.Image):
            if image_init.dtype != np.uint8:
                image_init = np.round(image_init)
                image_init = np.clip(image_init, 0, 255)
                image_init = image_init.astype(np.uint8)
            image_init = Image.fromarray(image_init)
        
        width, height = image_init.size
        if height != self.height_diffusion or width != self.width_diffusion:
            image_init = lt.resize(image_init, size=(self.height_diffusion, self.width_diffusion))
        self.image_init = image_init

    def set_embeddings(self, *args):
        """Sets the embeddings, accepting either four separate embeddings or a list of four:
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds"""
        if len(args) == 1 and isinstance(args[0], list) and len(args[0]) == 4:
            self.embeds = args[0]
        elif len(args) == 4:
            self.embeds = list(args)
        else:
            raise ValueError("Invalid input. Please provide either four separate embeddings or a list containing the four embeddings.")

    def generate(self):
        """Generates and returns an image diffusion."""
        assert self.embeds is not None, "Embeddings not set! Call set_embeddings first."
        kwargs = {}
        kwargs['latents'] = self.latents
        kwargs['num_inference_steps'] = self.num_inference_steps
        kwargs['guidance_scale'] = self.guidance_scale
        kwargs['prompt_embeds'] = self.embeds[0]
        kwargs['negative_prompt_embeds'] = self.embeds[1]
        kwargs['pooled_prompt_embeds'] = self.embeds[2]
        kwargs['negative_pooled_prompt_embeds'] = self.embeds[3]

        if self.use_image2image:
            assert self.image_init is not None, "Input image not set! Call set_input_image first."
            kwargs['image'] = self.image_init
            kwargs['strength'] = self.strength
            kwargs['image'] = self.image_init
            kwargs['strength'] = self.strength
            
        img_diffusion = self.pipe(**kwargs).images[0]
    
        return img_diffusion
    



if __name__ == "__main__experimental":
    # import util

    # from u_unet_modulated import forward_modulated
    from embeddings_mixer import EmbeddingsMixer
    # init latents
    pb.w = de.width_latents
    pb.h = de.height_latents
    latents = pb.get_latents()
    
    # noise img2img
    noise_img2img = torch.randn((1,4,de.height_latents,de.width_latents)).half().cuda() * 0
    
    # init embeds
    embeds_mod_full = pb.get_prompt_embeds("big round bear")
    embeds = {'prompt_embeds': embeds_mod_full[0],
              'negative_prompt_embeds': embeds_mod_full[1],
              'pooled_prompt_embeds': embeds_mod_full[2],
              'negative_pooled_prompt_embeds': embeds_mod_full[3]}
    
    # image
    import requests
    from PIL import Image
    from io import BytesIO
    import numpy as np
    
    url = "https://cdn.britannica.com/79/232779-050-6B0411D7/German-Shepherd-dog-Alsatian.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    image_init = np.array(img)
        
    modulations = {}
    modulations_noise = {}
    for i in range(3):
        modulations_noise[f'e{i}'] = util.get_noise_for_modulations(util.get_sample_shape_unet(f'e{i}', de.height_latents, de.width_latents), de.pipe_text2img)
        modulations_noise[f'd{i}'] = util.get_noise_for_modulations(util.get_sample_shape_unet(f'd{i}', de.height_latents, de.width_latents), de.pipe_text2img)
        
    modulations_noise['b0'] = util.get_noise_for_modulations(util.get_sample_shape_unet('b0', de.height_latents, de.width_latents), de.pipe_text2img)
    modulations['modulations_noise'] = modulations_noise
        
    cross_attention_kwargs ={}
    cross_attention_kwargs['modulations'] = modulations        
        
    kwargs = {}
    kwargs['guidance_scale'] = 0.0
    kwargs['latents'] = latents
    kwargs['prompt_embeds'] = embeds['prompt_embeds']
    kwargs['negative_prompt_embeds'] = embeds['negative_prompt_embeds']
    kwargs['pooled_prompt_embeds'] = embeds['pooled_prompt_embeds']
    kwargs['negative_pooled_prompt_embeds'] = embeds['negative_pooled_prompt_embeds']
    kwargs['strength'] = 0.5
    kwargs['noise_img2img'] = noise_img2img
    
    if len(cross_attention_kwargs) > 0:
        kwargs['cross_attention_kwargs'] = cross_attention_kwargs
    
    img_output = de.generate_image(kwargs)
    
    plt.imshow(np.array(img_output))
    

    
if __name__ == '__main__':
    de_txt = DiffusionEngine(use_image2image=False, height_diffusion_desired=700, width_diffusion_desired=1024)
    em = EmbeddingsMixer(de_txt.pipe)
    embeds = em.encode_prompt("photo of a house")
    de_txt.set_embeddings(embeds)
    img_init = de_txt.generate()
    
    de_img = DiffusionEngine(use_image2image=True, height_diffusion_desired=512, width_diffusion_desired=512)
    embeds = em.encode_prompt("blue painting of a house")
    de_img.set_embeddings(embeds)
    de_img.set_input_image(img_init)
    de_img.generate()
    
    
