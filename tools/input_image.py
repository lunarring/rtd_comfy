import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from .wobblers import WobbleMan
from .segmentation_detection import HumanSeg
from lunar_tools import exception_handler
import lunar_tools as lt

def img2tensor(tensor):
    """
    Converts a tensor to a numpy array.

    Parameters:
    tensor (torch.Tensor): The input tensor to be converted.

    Returns:
    np.ndarray: The converted numpy array.
    """
    return (tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()) / 255.0

def tensor2image(input_data):
    """
    Converts a tensor to a numpy array.

    Parameters:
    input_data (torch.Tensor): The input tensor to be converted. It should be in the format (C, H, W) where
                                C is the number of channels, H is the height, and W is the width.

    Returns:
    np.ndarray: The converted numpy array or the input if it is not a tensor.
    """
    
    # Check if the input is a tensor
    if not isinstance(input_data, (torch.Tensor)):
        return input_data
    
    # Ensure the tensor is on the CPU and convert to a numpy array
    converted_data = input_data.cpu().numpy() if input_data.is_cuda else input_data.numpy()
    if len(converted_data.shape) == 4:
        converted_data = converted_data[0, :, :, :]
    converted_data = np.clip(converted_data * 255, 0, 255)
    return converted_data

def zoom_image_torch(input_tensor, zoom_factor):
    try:
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
    except Exception as e:
        print(f"zoom_image_torch failed! {e}. returning original input")
        return input_tensor


# grid resampler
def torch_resample(tex, grid,padding_mode="reflection", mode='bilinear'):
#    import pdb; pdb.set_trace()
    if len(tex.shape) == 3:                        # add singleton to batch dim
        return F.grid_sample(tex.view((1,)+tex.shape),grid.view((1,)+grid.shape),padding_mode=padding_mode,mode=mode)[0,:,:,:].permute([1,2,0])
    elif len(tex.shape) == 4:
        return F.grid_sample(tex,grid.view((1,)+grid.shape),padding_mode=padding_mode,mode=mode)[0,:,:,:].permute([1,2,0])
    else:
        raise ValueError('torch_resample: bad input dims')
        
def torch_rotate(x, a):
    theta = torch.zeros((1,2,3)).cuda(x.device)
    
    theta[0,0,0] = np.cos(a)
    theta[0,0,1] = -np.sin(a)
    theta[0,1,0] = np.sin(a)
    theta[0,1,1] = np.cos(a)
    
    basegrid = F.affine_grid(theta, (1,2,x.shape[1], x.shape[2]))[0,:,:,:]
    return torch_resample(x.unsqueeze(0), basegrid)  


class InputImageProcessor():
    def __init__(self, do_human_seg=True, do_blur=True, blur_kernel=3, is_infrared=False):
        self.brightness = 1.0
        self.saturization = 1.0
        self.hue_rotation_angle = 0
        self.blur_kernel = None
        self.resizing_factor_humanseg = 0.4 # how much humanseg img is downscaled internally, makes things faster.
        
        # human body segmentation
        self.human_seg = HumanSeg(resizing_factor=self.resizing_factor_humanseg)
        self.set_blur_size(self.blur_kernel)
        
        self.do_human_seg = do_human_seg
        self.do_blur = do_blur
        self.is_infrared = is_infrared
        self.flip_axis = None
        
        self.list_history_frames = []
        
        
    def set_resizing_factor_humanseg(self, resizing_factor):
        self.resizing_factor_humanseg = resizing_factor
        self.human_seg.set_resizing_factor(resizing_factor)

    def set_brightness(self, brightness=1):
        self.brightness = brightness

    def set_saturization(self, saturization):
        self.saturization = saturization
        
    def set_hue_rotation(self, hue_rotation_angle=0):
        self.hue_rotation_angle = hue_rotation_angle
        
    def set_blur_size(self, blur_kernel):
        if blur_kernel != self.blur_kernel:
            self.blur = lt.MedianBlur((blur_kernel, blur_kernel))
        
    def set_blur(self, do_blur=True):
        self.do_blur = do_blur
        
    def set_infrared(self, is_infrared=True):
        self.is_infrared = is_infrared
        
    def set_human_seg(self, do_human_seg=True):
        self.do_human_seg = do_human_seg

    def set_flip(self, do_flip, flip_axis=1):
        if do_flip:
            self.flip_axis = flip_axis
        else:
            self.flip_axis = None

        
    # @exception_handler
    def process(self, img):

        if isinstance(img, torch.Tensor):
            img = img.squeeze(0)
            img = img.cpu().numpy()
            img = np.asarray(255*img, dtype=np.uint8)

        if self.flip_axis is not None:
            img = np.flip(img, axis=self.flip_axis)
        
        if self.do_blur:
            img_torch = torch.from_numpy(img.copy()).to(0).float()
            img = self.blur(img_torch.permute([2,0,1])[None])[0].permute([1,2,0]).cpu().numpy()
        
        # human body segmentation mask
        if self.do_human_seg:
            human_segmmask = self.human_seg.get_mask(img)
            img = self.human_seg.apply_mask(img)
        else:
            human_segmmask = None
        
        # adjust brightness
        img = img.astype(np.float32)
        
        # if infrared, take mean of RGB channels and place it into red channel
        # the image can be then color-rotated with hue adjustments to fit the prompt color space
        if self.is_infrared:
            mean_intensity = img.mean(2)
            img[:,:,0] = mean_intensity
            img[:,:,1:] = 0
        
        # # time-averaging
        # self.list_history_frames.append(img)
        # if len(self.list_history_frames) > 10:
        #     self.list_history_frames = self.list_history_frames[1:]
        #     img = np.mean(np.stack(self.list_history_frames), axis=0)
        
        img *= self.brightness
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

        # convert the image to HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
        # adjust saturization
        img_hsv[:, :, 1] *= self.saturization

        # Rotate the hue
        # Hue is represented in OpenCV as a value from 0 to 180 instead of 0 to 360...
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + (self.hue_rotation_angle / 2)) % 180

        # clip the values to stay in valid range
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
        # convert the image back to BGR
        img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        if human_segmmask is not None:
            human_segmmask *= 255
            human_segmmask = np.repeat(np.expand_dims(human_segmmask, 2), 3, axis=2)
            human_segmmask = human_segmmask.astype(np.uint8)
        
        return img, human_segmmask


class AcidProcessor():
    def __init__(self, device='cuda', 
                 height_diffusion=576, 
                 width_diffusion=1024):
        self.device = device
        self.last_diffusion_image_torch = None
        self.width_diffusion = width_diffusion
        self.height_diffusion = height_diffusion
        
        self.wobbleman = WobbleMan(device)
        self.wobbleman.init('a01')

        self.acid_strength = 0.05
        self.acid_strength_foreground = 0.01
        self.coef_noise = 0.15
        self.x_shift = 0
        self.y_shift = 0
        self.zoom_factor = 1
        self.rotation_angle = 0
        self.do_acid_tracers = False
        self.apply_humansegm_mask = False
        self.do_acid_wobblers = False
        self.do_flip_invariance = False
        self.human_segmmask = None
        self.wobbler_control_kwargs = {}
        self.flip_state = 0
        self.stereo_scaling_applied = False

    def set_wobbler_control_kwargs(self, wobbler_control_kwargs):
        self.wobbler_control_kwargs = wobbler_control_kwargs

    def set_acid_strength(self, acid_strength):
        self.acid_strength = acid_strength

    def set_acid_strength_foreground(self, acid_strength_foreground):
        self.acid_strength_foreground = acid_strength_foreground

    def set_coef_noise(self, coef_noise):
        self.coef_noise = coef_noise

    def set_x_shift(self, x_shift):
        self.x_shift = x_shift

    def set_y_shift(self, y_shift):
        self.y_shift = y_shift

    def set_zoom_factor(self, zoom_factor):
        self.zoom_factor = zoom_factor

    def set_rotation_angle(self, rotation_angle):
        self.rotation_angle = rotation_angle

    def set_do_acid_tracers(self, do_acid_tracers):
        self.do_acid_tracers = do_acid_tracers

    def set_apply_humansegm_mask(self, apply_humansegm_mask):
        self.apply_humansegm_mask = apply_humansegm_mask

    def set_do_acid_wobblers(self, do_acid_wobblers):
        self.do_acid_wobblers = do_acid_wobblers

    def set_human_segmmask(self, human_segmmask):
        self.human_segmmask = human_segmmask

    def set_flip_invariance(self, do_flip_invariance):
        self.do_flip_invariance = do_flip_invariance

    def set_stereo_image(self, do_stereo_image=False):
        if not self.stereo_scaling_applied:
            self.height_diffusion = self.height_diffusion * 2
            self.stereo_scaling_applied = True
    
    # @exception_handler
    def process(self, image_input):
        if isinstance(image_input, torch.Tensor):
            image_input = image_input.squeeze(0)
            image_input = image_input.cpu().numpy()
            image_input = np.asarray(255*image_input, dtype=np.uint8)
        if self.last_diffusion_image_torch is None:
            print("InputImageProcessor: last_diffusion_image_torch=None. returning original image...")
            return image_input
        
        last_diffusion_image_torch = self.last_diffusion_image_torch
        width_diffusion = self.width_diffusion
        height_diffusion = self.height_diffusion
        
        # acid transform
        # wobblers
        if self.do_acid_wobblers:
            required_keys = ['amp', 'frequency', 'edge_amp']
            wobbler_control_kwargs_are_good = all(key in self.wobbler_control_kwargs for key in required_keys)
            if not wobbler_control_kwargs_are_good:
                print("Some keys are missing in wobbler_control_kwargs. Required keys are: ", required_keys)
            else:
                resample_grid = self.wobbleman.do_acid(last_diffusion_image_torch, self.wobbler_control_kwargs)
                last_diffusion_image_torch = torch_resample(last_diffusion_image_torch.permute([2,0,1]), ((resample_grid * 2) - 1)).float()
        
        # zoom
        if self.zoom_factor != 1 and self.zoom_factor > 0:
            last_diffusion_image_torch = zoom_image_torch(last_diffusion_image_torch, self.zoom_factor)
            
        # rotations
        if self.rotation_angle != 0:
            padding = int(last_diffusion_image_torch.shape[1] // (2*np.sqrt(2)))
            padding = (padding, padding)
            last_diffusion_image_torch = transforms.Pad(padding=padding, padding_mode='reflect')(last_diffusion_image_torch.permute(2,0,1))
            last_diffusion_image_torch = transforms.functional.rotate(last_diffusion_image_torch, angle=self.rotation_angle, interpolation=transforms.functional.InterpolationMode.BILINEAR, expand=False).permute(1,2,0)
            last_diffusion_image_torch = last_diffusion_image_torch[padding[0]:last_diffusion_image_torch.shape[0]-padding[0],padding[1]:last_diffusion_image_torch.shape[1]-padding[1]]            
        
        # acid plane translations
        if self.x_shift != 0 or self.y_shift != 0:
            last_diffusion_image_torch = torch.roll(last_diffusion_image_torch, (self.y_shift, self.x_shift), (0,1))

        img_input_torch = torch.from_numpy(image_input.copy()).to(self.device).float()
        if img_input_torch.shape[0] != height_diffusion or img_input_torch.shape[1] != width_diffusion:
            img_input_torch = lt.resize(img_input_torch.permute((2,0,1)), size=(height_diffusion, width_diffusion)).permute((1,2,0))
        
        if self.apply_humansegm_mask:
            if len(self.human_segmmask.shape) == 3:
                human_segmmask = self.human_segmmask[:,:,0]/255
            else:
                human_segmmask = self.human_segmmask
                
            human_segmmask_resized = np.expand_dims(cv2.resize(human_segmmask, (width_diffusion, height_diffusion)),2)
            human_segmmask_torch = torch.from_numpy(human_segmmask_resized).to(self.device)
        
        if self.do_acid_tracers and self.apply_humansegm_mask:
            img_input_torch_current = img_input_torch.clone()
            img_input_torch = (1.-self.acid_strength)*img_input_torch + self.acid_strength*last_diffusion_image_torch
            img_input_torch = human_segmmask_torch*img_input_torch_current + (1-human_segmmask_torch)*img_input_torch
            img_input_torch = (1.-self.acid_strength_foreground)*img_input_torch + self.acid_strength_foreground*last_diffusion_image_torch
        else:
            img_input_torch = (1.-self.acid_strength)*img_input_torch + self.acid_strength*last_diffusion_image_torch
            
            
        # additive noise
        if self.coef_noise > 0:
            torch.manual_seed(420)
            t_rand = (torch.randn(img_input_torch.shape, device=img_input_torch.device)) * self.coef_noise * 20
            if self.apply_humansegm_mask:
                t_rand *= (1-human_segmmask_torch)
            img_input_torch += t_rand
        
        img_input = img_input_torch.cpu().numpy()
        return img_input
    
    def update(self, img_diffusion):
        self.last_diffusion_image_torch = torch.from_numpy(img_diffusion).to(self.device, dtype=torch.float)

    
if __name__ == '__main__':
    acid_process = InputImageProcessor()
    
