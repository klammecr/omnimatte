# Third Party
import torch
import torch.nn as nn
import numpy as np

# In House
from utils import transform2h, apply_transform

class HomographyManager:
    def __init__(self, hom_file, n_images, transform_params):
        """
        Read homography file and set up homography data.
        """

        with open(hom_file) as f:
            h_data = f.readlines()
        h_scale = h_data[0].rstrip().split(' ')
        self.h_scale_x = int(h_scale[1])
        self.h_scale_y = int(h_scale[2])
        h_bounds = h_data[1].rstrip().split(' ')
        self.h_bounds_x = [float(h_bounds[1]), float(h_bounds[2])]
        self.h_bounds_y = [float(h_bounds[3]), float(h_bounds[4])]
        homographies = h_data[2:2 + n_images]
        homographies = [torch.from_numpy(np.array(line.rstrip().split(' ')).astype(np.float32).reshape(3, 3)) for line
                        in
                        homographies]
        
        # Set member variables
        self.homographies     = homographies
        self.transform_params = transform_params
        self.hom_delta        = nn.Parameter(torch.zeros(n_images, 3, 3))


    def get_background_uv(self, idx, w, h):
        """
        Return background layer UVs at 'index' (output range [0, 1]).
        This is the UV from the reference frame to the indexed frame
        """
        ramp_u = torch.linspace(0, 1, steps=w).unsqueeze(0).repeat(h, 1)
        ramp_v = torch.linspace(0, 1, steps=h).unsqueeze(-1).repeat(1, w)
        ramp = torch.stack([ramp_u, ramp_v], 0)
        if self.homographies is not None:
            H = self.homographies[idx]

            # scale to [0, orig width/height]
            ramp[0] *= self.h_scale_x
            ramp[1] *= self.h_scale_y

            # apply homography
            ramp = ramp.reshape(2, -1)  # [2, H, W]
            [xt, yt] = transform2h(ramp[0], ramp[1], torch.inverse(H))

            # scale from world to [0,1]
            xt -= self.h_bounds_x[0]
            xt /= (self.h_bounds_x[1] - self.h_bounds_x[0])
            yt -= self.h_bounds_y[0]
            yt /= (self.h_bounds_y[1] - self.h_bounds_y[0])

            # restore shape
            ramp = torch.stack([xt.reshape(h, w), yt.reshape(h, w)], 0)

        return ramp
    
    def get_background_flow(self, index, w, h, opt_width, opt_height):
        """
        Return background layer UVs at 'index' (output range [0, 1]).
        """
        if index >= len(self.homographies):
            return torch.zeros(2, h, w)
        
        if self.homographies is not None:
            ramp_u = torch.linspace(0, self.h_scale_x, steps=w).unsqueeze(0).repeat(h, 1)
            ramp_v = torch.linspace(0, self.h_scale_y, steps=h).unsqueeze(-1).repeat(1, w)
            ramp_ = torch.stack([ramp_u, ramp_v], 0)
            ramp = ramp_.reshape(2, -1)

            # Find relative homographies
            H_0      = self.homographies[index]
            H_1      = self.homographies[index + 1]
            final_H  = (H_1 @ torch.inverse(H_0)) + self.hom_delta[index]
            [xt, yt] = transform2h(ramp[0], ramp[1], final_H)
            # [xt, yt] = transform2h(ramp[0], ramp[1], torch.inverse(H_0))
            # [xt, yt] = transform2h(xt, yt, H_1)

            # restore shape
            flow = torch.stack([xt.reshape(h, w), yt.reshape(h, w)], 0)
            flow -= ramp_
            # # scale from world to [-1, 1]
            # flow[0] /= .5 * self.h_scale_x
            # flow[1] /= .5 * self.h_scale_y

            # scale from world to image space
            flow[0] *= opt_width / self.h_scale_x
            flow[1] *= opt_height / self.h_scale_y
        else:
            flow = torch.zeros(2, h, w)
        return flow