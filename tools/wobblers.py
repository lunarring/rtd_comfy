import os, sys
import numpy as np
import torch
import time

#%%
def get_cartesian_resample_grid(shape_hw, gpu, use_half_precision=False):
    # initialize reesampling Cartesian grid
    theta = torch.zeros((1,2,3)).cuda(gpu)
    theta[0,0,0] = 1
    theta[0,1,1] = 1
    
    basegrid = F.affine_grid(theta, (1, 2, shape_hw[0], shape_hw[1]))
    iResolution = torch.tensor([shape_hw[0], shape_hw[1]]).float().cuda(gpu).unsqueeze(0).unsqueeze(0)

    grid = (basegrid[0,:,:,:] + 1) / 2
    grid *= iResolution
    cartesian_resample_grid = grid / iResolution

    if use_half_precision:
        cartesian_resample_grid = cartesian_resample_grid.half()
        
    return cartesian_resample_grid

def get_kernel_cdiff(gpu=None):
    g = np.ones((3,3))
    g[1,1] = -8
    return get_kernel(g, gpu)

def apply_kernel(img, kernel, nmb_repetitions=1):
    if len(img.shape)==2:
        img = img.expand(1,1,img.shape[0],img.shape[1])
    else:
        print("WARNING! 3D NEEDS 3D KERNEL!")
        img = img.permute([2,0,1])
        img = img.expand(1,img.shape[0],img.shape[1],img.shape[2])
        
    for i in range(nmb_repetitions):
        img = kernel(img)
        
    return img.squeeze()

def get_kernel(kernel_weights, gpu=None):
    assert gpu is not None, "pythoniac maniac gpuiac"
    assert len(kernel_weights.shape) == 2, '2D conv!'
    assert kernel_weights.shape[0] == kernel_weights.shape[1], 'square!'
    padding = int((kernel_weights.shape[0]-1) / 2)
    m = torch.nn.Conv2d(1, 1, kernel_weights.shape[0], padding=padding, stride=1)
    m.bias[0] = 0
    m.weight[0,0,:,:] = torch.from_numpy(kernel_weights.astype(np.float32))
    m = m.cuda(gpu)
    return m

def get_kernel_gauss(ksize=3, gpu=None):
    x, y = np.meshgrid(np.linspace(-1,1,ksize), np.linspace(-1,1,ksize))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    g = g/np.sum(g)
    return get_kernel(g, gpu)

class TimeMan():
    def __init__(self, use_counter=False, count_time_increment=0.02):
        self.t_start = time.perf_counter()
        self.t_last = self.t_start 
        self.use_counter = use_counter
        self.count_time_increment = count_time_increment
        self.t_last_interval = 0
        self.dt_interval=-1
        
    def set_interval(self, dt_inverval=1):
        self.dt_interval = dt_inverval
        self.t_interval_last = time.perf_counter()
        
    def reset_start(self):
        self.t_start = time.perf_counter()
        self.t_last = self.t_start       
        
    def check_interval(self):
        self.update()
        if self.t_last > self.t_interval_last + self.dt_interval:
            self.t_interval_last = np.copy(self.t_last)
            # print("self.t_last: {} self.t_inverval_last {}".format(self.t_last, self.t_interval_last))
            return True
        else:
            return False
    
    def tic(self):
        self.reset_start()
        
    def toc(self):
        self.update()
        dt = self.t_last - self.t_start
        print("dt = {}ms".format(int(1000*dt)))
    
    def get_time(self):
        return self.t_last
        
    def get_dt(self):
        if not self.use_counter:
            self.t_last = time.perf_counter()
        dt = self.t_last - self.t_start
        return dt

    def update(self):
        if self.use_counter:
            self.t_last += self.count_time_increment 
        else:
            self.t_last = time.perf_counter()
            
class WobbleMan():
    def __init__(self, gpu, time_man=None):
        self.gpu = gpu
        self.do_acid = self.do_acid_not_init
        
        if time_man is None:
            self.tm = TimeMan()
        else:
            self.tm = time_man
        
        with torch.no_grad():
            self.cartesian_resample_grid = None
            self.gkernel = get_kernel_gauss(gpu=gpu)
            self.ckernel = get_kernel_cdiff(gpu=gpu)
        self.ran_once = False
        
        
    def do_acid_not_init(self):
        print("ACID NOT INITIALIZED! run e.g. init_j01")
        
    
    def get_resample_grid(self, shape_hw):
        resample_grid = get_cartesian_resample_grid(shape_hw, self.gpu)
        return resample_grid

    
    def init(self, profile_name='j01'):
        init_function = "init_{}".format(profile_name)
        assert hasattr(self, init_function), "acid_man: unknown profile! {} not found".format(init_function)
        
        self.init_function = getattr(self, init_function)
        self.init_function()
        acid_function = "do_acid_{}".format(profile_name)
        assert hasattr(self, acid_function), "acid_man: unknown profile! {} not found".format(acid_function)
        self.do_acid = getattr(self, acid_function)
        

    def init_a01(self):
        self.phase_rot = 0
        self.freq_rot = 0
        self.is_fluid = False
        
    def do_acid_a01(self, source, control_kwargs):
        
        amp = control_kwargs['amp']
        frequency = control_kwargs['frequency']
        edge_amp = control_kwargs['edge_amp']
        
        dt = self.tm.get_dt()
        frame_sum = torch.sum(source, axis=2)
        
        frame_sum -= frame_sum.min()
        frame_sum /= frame_sum.max()
        
        edges = apply_kernel(frame_sum, self.ckernel)
    
        # where
        edges = edges.abs()
        edges /= edges.max()
        
        # factor = 30
        factor = 1
        edges = apply_kernel(edges, self.gkernel, factor)
        edges /= edges.max()
        
        edges = 1 - edges
        edges *= edge_amp
        
        # which phase
        factor = int(1)
        fsum_amp = apply_kernel(frame_sum, self.gkernel, factor)
        fsum_amp -= fsum_amp.min()
        fsum_amp /= fsum_amp.max()
        fsum_amp *= 2*np.pi
        
        # xy modulatioself.nS: frequency
        if frequency != self.freq_rot:
            self.phase_rot += dt*(self.freq_rot - frequency)
            self.freq_rot = frequency    
        
        y_wobble = torch.sin(dt * self.freq_rot  + fsum_amp + self.phase_rot)
        x_wobble = torch.cos(dt * self.freq_rot  + fsum_amp + self.phase_rot)    
        
        v_edges = edges * y_wobble * amp
        h_edges = edges * x_wobble * amp
        
        shape_hw = source.shape
        resample_grid = self.get_resample_grid(shape_hw)
        self.identity_resample_grid = resample_grid.clone()
        resample_grid[:,:,0] += v_edges
        resample_grid[:,:,1] += h_edges
        
        return resample_grid
    
    

                
                