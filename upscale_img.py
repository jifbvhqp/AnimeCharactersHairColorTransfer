import argparse
import cv2
import glob
import math
import numpy as np
from PIL import Image
import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from torch.nn import functional as F
import matplotlib.pyplot as plt
class RealESRGANer():
    def __init__(self, scale, model_path, tile=0, tile_pad=10, pre_pad=10):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        self.model = model.to(self.device)

    def pre_process(self, img, changedim = True):
        if changedim:
            img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            self.img = img.unsqueeze(0).to(self.device)
        else:
            self.img = img.to(self.device)

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        try:
            # inference
            with torch.no_grad():
                self.output = self.model(self.img)
        except Exception as error:
            print('Error', error)

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

model_path = 'upscale_model/RealESRGAN_x4plus.pth'
scale = 4
suffix = 'out'
tile = 0
tile_pad = 10
pre_pad = 0
alpha_upsampler = 'realesrgan'
ext = 'auto'

def upscale_imgPath(path = 'images.jpg'):
    upsampler = RealESRGANer(scale=scale, model_path=model_path, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad)
    im = Image.open(path)  
    im = np.array(im)
    if np.max(im) > 255:  # 16-bit image
        max_range = 65535
        print('\tInput is a 16-bit image')
    else:
        max_range = 255
    im = im / max_range
    plt.imshow(im)
    plt.show()
    upsampler.pre_process(im)
    upsampler.process()
    output_img = upsampler.post_process()
    output_img = output_img.data.squeeze().permute(1,2,0).float().cpu().clamp_(0, 1).numpy()
    output = (output_img * float(max_range)).round().astype(np.uint8)
    print(output.shape)
    plt.imshow(output)
    plt.show()

def upscale_tensor(im):
    upsampler = RealESRGANer(scale=scale, model_path=model_path, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad)
    upsampler.pre_process(im, changedim = False)
    upsampler.process()
    output_img = upsampler.post_process()
    output_img = output_img.data.float().cpu().clamp_(0, 1)
    return output_img

