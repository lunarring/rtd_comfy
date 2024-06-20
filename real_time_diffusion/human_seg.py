#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class HumanSeg():
    """
    This class is used for human segmentation in images. It uses a pre-trained model from PyTorch's model zoo 
    (deeplabv3_resnet101 by default) to predict the human body in the image and generate a mask. The mask can be 
    used for various applications like background removal, human pose estimation etc.
    
    Attributes:
        model: The pre-trained model used for human segmentation. available models: deeplabv3_resnet50 / deeplabv3_resnet101 / deeplabv3_mobilenet_v3_large
        preprocess: The preprocessing transformations applied on the input image before feeding it to the model.
    """
    # 
    def __init__(self, model_name='deeplabv3_resnet101'):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        
        self.model.eval()
        self.model.to('cuda')
        
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def get_mask(self, cam_img):
        sel_id = 15 # human body code
        
        input_image = Image.fromarray(cam_img)
        input_image = input_image.convert("RGB")
        
        input_tensor = self.preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        
        # move the input and model to GPU for speed if available
        input_batch = input_batch.to('cuda')
        
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        mask = output_predictions.byte().cpu().numpy()
        
        mask = (mask == sel_id).astype(np.uint8)
        
        return mask

if __name__ == '__main__':
    import lunar_tools as lt
    import matplotlib.pyplot as plt
    shape_cam=(300,400) 
    cam = lt.WebCam(cam_id=-1, shape_hw=shape_cam)
    
    human_seg = HumanSeg()
    while True:
        cam_img = cam.get_img()
        cam_img = np.flip(cam_img, axis=1)    
    
        mask = human_seg.get_mask(cam_img)
        
        plt.imshow(mask); plt.ion(); plt.show()        
