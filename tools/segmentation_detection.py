#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
import time
import lunar_tools as lt

class HumanSeg:
    """
    This class is used for human segmentation in images. It uses a pre-trained model from PyTorch's model zoo 
    (deeplabv3_resnet101 by default) to predict the human body in the image and generate a mask. The mask can be 
    used for various applications like background removal, human pose estimation etc.
    
    Attributes:
        model: The pre-trained model used for human segmentation. available models: deeplabv3_resnet50 / deeplabv3_resnet101 / deeplabv3_mobilenet_v3_large
        size (tuple, optional): The desired size (height, width) for the output tensor. If provided, this overrides the downscaling_factor.
        resizing_factor (float, optional): The factor by which to downscale the input tensor. Defaults to None. Ignored if size is provided.
        preprocess: The preprocessing transformations applied on the input image before feeding it to the model.
    """
    # 
    def __init__(self, model_name='deeplabv3_resnet101', resizing_factor=None, size=None):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        
        self.model.eval()
        self.model.to('cuda')
        
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.sel_id = 15 # human body code
        self.resizing_factor = resizing_factor
        self.size = size
        self.mask = None

    def set_resizing_factor(self, resizing_factor):
        """
        This method sets the resizing factor for the HumanSeg class.

        Args:
            resizing_factor (float): The factor by which to downscale the input tensor.
        """
        self.resizing_factor = resizing_factor

        
    def get_mask(self, input_img):
        """
        This method generates a binary mask for the human body in the given image. The mask can be used for various applications like background removal, human pose estimation etc.

        Args:
            cam_img (np.ndarray): The input image in which the human body is to be segmented. The image should be in RGB format.

        Returns:
            sets self.mask, a binary mask for the human body in the input image. The mask is of the same size as the input image.
        """
        if isinstance(input_img, np.ndarray) and input_img.dtype == np.float32:
            input_img = (input_img).astype(np.uint8)
        input_image = Image.fromarray(input_img)
        input_image = input_image.convert("RGB")
        
        input_tensor = self.preprocess(input_image)
        if self.resizing_factor is not None or self.size is not None:
            size_orig = input_image.size
            input_tensor = lt.resize(input_tensor, resizing_factor=self.resizing_factor, size=self.size)

        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        
        # move the input and model to GPU for speed if available
        input_batch = input_batch.to('cuda')
        
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        mask = output_predictions.byte().cpu().numpy()

        mask = (mask == self.sel_id).astype(np.uint8)

        if self.resizing_factor is not None or self.size is not None:
            mask = lt.resize(mask, size=(input_image.size[1], input_image.size[0]))
            mask = np.round(mask)
            mask = np.where(mask > 0.5, 1, 0)
            
        # cumulative mask
        do_cumulative_mask = False
        if do_cumulative_mask:
            if self.mask is not None:
                self.mask = self.mask + mask
                self.mask[self.mask > 1] = 1
            else:
                self.mask = mask
        else:
            self.mask = mask

        return mask


    def apply_mask(self, input_img, mask_strength=1, invert_mask=False):
        """
        This method applies the previously generated mask to the input image.

        Args:
            input_img (np.ndarray): The input image to which the mask is to be applied. The image should be in RGB format.
            invert_mask (bool): If set to True, the mask will be inverted before being applied.

        Returns:
            np.ndarray: The masked image.
        """
        if self.mask is None:
            raise ValueError("No mask has been generated. Please call get_mask() first.")
        is_pil = False
        if isinstance(input_img, Image.Image):
            is_pil = True
            input_img = np.array(input_img)
        else:
            if input_img.dtype == np.float32:
                input_img = (input_img).astype(np.uint8)
        assert input_img.shape[:2] == self.mask.shape, "The input image and the mask must have the same dimensions."
        assert 0 <= mask_strength <= 1, "mask_strength should be between 0 and 1"
        
        mask = 1 - self.mask if invert_mask else self.mask
        masked_img = input_img * np.expand_dims(mask, axis=2)
        
        if mask_strength != 1:
            input_img = input_img.astype(float)
            masked_img = masked_img.astype(float)
            masked_img = (1-mask_strength)*input_img + mask_strength * masked_img
            masked_img = np.round(masked_img)
            masked_img = np.clip(masked_img, 0, 255)
            masked_img = masked_img.astype(np.uint8)

        return masked_img

    


if __name__ == '__main__':
    import lunar_tools as lt
    import matplotlib.pyplot as plt
    
    shape_cam=(300,400) 
    cam = lt.WebCam(cam_id=-1, shape_hw=shape_cam)
    
    human_seg = HumanSeg(resizing_factor=1.0)
    
    #%%
    while True:
        cam_img = cam.get_img()
        cam_img = np.flip(cam_img, axis=1)    
    
        human_seg.get_mask(cam_img)
        img = human_seg.apply_mask(cam_img)
        
        plt.imshow(img); plt.ion(); plt.show()        
