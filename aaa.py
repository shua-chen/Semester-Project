import cv2
import math
import random
import numpy as np
import os.path as osp
import os
from PIL import Image
from prepare_data import degradations as degradations
from prepare_data.utils import setup_logger
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, 
                                        adjust_hue, adjust_saturation, normalize)



def color_jitter(img, shift):
    """jitter color: randomly jitter the RGB values, in numpy formats"""
    jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
    img = img + jitter_val
    img = np.clip(img, 0, 1)
    return img


def color_jitter_pt(img, brightness, contrast, saturation, hue):

    """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
    fn_idx = torch.randperm(4)
    for fn_id in fn_idx:
        if fn_id == 0 and brightness is not None:
            brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
            img = adjust_brightness(img, brightness_factor)

        if fn_id == 1 and contrast is not None:
            contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
            img = adjust_contrast(img, contrast_factor)

        if fn_id == 2 and saturation is not None:
            saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
            img = adjust_saturation(img, saturation_factor)

        if fn_id == 3 and hue is not None:
            hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
            img = adjust_hue(img, hue_factor)
    return img

def get_lq_img(opt,dataset):

    kernel_list=opt['kernel_list']
    kernel_prob=opt['kernel_prob']
    blur_kernel_size=opt['blur_kernel_size']
    blur_sigma=opt['blur_sigma']
    downsample_range=opt['downsample_range']
    noise_range=opt['noise_range']
    jpeg_range=opt['jpeg_range']
    color_jitter_prob=opt.get('color_jitter_prob',None)
    color_jitter_pt_prob=opt.get('color_jitter_pt_prob',None)
    color_jitter_shift=opt.get('color_jitter_shift',None)
    gray_prob=opt.get('gray_prob',None)
    if color_jitter_shift is not None:
        color_jitter_shift /= 255.
        
    def degrad(img):
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5 if opt['use_hflip'] else 0),  
            transforms.RandomVerticalFlip(p=0.5 if opt['use_vflip'] else 0),   
            transforms.RandomRotation(degrees=opt.get('rotate', 0)),   
        ])
        img_gt = trans(img)

        # Convert image to numpy array
        img_gt = np.array(img_gt).astype(np.float32)/255.0        
        
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = degradations.random_mixed_kernels(
            kernel_list,
            kernel_prob,
            blur_kernel_size,
            blur_sigma,
            blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)

        # downsample
        scale = np.random.uniform(downsample_range[0], downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, noise_range)
        # jpeg compression
        if jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if color_jitter_prob is not None and (np.random.uniform() < color_jitter_prob):
            img_lq = color_jitter(img_lq, color_jitter_shift)
        # random to gray (only for lq)
        if gray_prob is not None and np.random.uniform() < gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            

        # BGR to RGB, HWC to CHW, numpy to tensor
        #img_lq = torch.from_numpy(img_lq).permute(2, 0, 1).float()

        # random color jitter (pytorch version) (only for lq)
        if color_jitter_pt_prob is not None and (np.random.uniform() < color_jitter_pt_prob):
            brightness = opt.get('brightness', (0.5, 1.5))
            contrast = opt.get('contrast', (0.5, 1.5))
            saturation = opt.get('saturation', (0, 1.5))
            hue = opt.get('hue', (-0.1, 0.1))
            img_lq = color_jitter_pt(img_lq, brightness, contrast, saturation, hue)

        if opt.get('gt_gray'):  # whether convert GT to gray images
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])  # repeat the color channels

        # Convert pixel values to the range 0-255 and clip values outside this range
        img_gt = np.clip((img_gt * 255).round(), 0, 255).astype(np.uint8)
        # BGR to RGB, numpy to PIL Image
        img_gt = Image.fromarray(img_gt).convert("RGB")
        # round and clip
        #img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255)/255.0
        #img_lq = transforms.ToPILImage()(img_lq).convert("RGB")
        img_lq = np.clip((img_lq * 255).round(), 0, 255).astype(np.uint8)
        img_lq = Image.fromarray(img_lq).convert("RGB")
        # Convert img_gt and img_lq to PIL Image format and ensure they are in RGB mode
        

        gt_transform=transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  
        ])
        lq_transform=transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),  
        ])
        img_gt=gt_transform(img_gt)
        img_lq=lq_transform(img_lq)
        # return shape 2*c*h*w
        stack_imgs=torch.stack([img_gt, img_lq])
        return stack_imgs
    def process_dataset(examples):
        new_images=[degrad(exam_image) for exam_image in examples['image']]
        # list2tensor
        new_images=torch.stack(new_images)
        examples['image'],examples['lq_image']=torch.chunk(new_images, 2, dim=1)
        examples['image']=torch.squeeze(examples['image'], dim=1)
        examples['lq_image']=torch.squeeze(examples['lq_image'], dim=1)
        examples['image']=list(torch.unbind(examples['image'], dim=0))     
        examples['lq_image']=list(torch.unbind(examples['lq_image'], dim=0))   
        return examples

    dataset=dataset.with_transform(process_dataset)
    return dataset