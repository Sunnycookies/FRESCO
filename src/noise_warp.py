import torch
import numpy as np
from PIL import Image
from src.utils import *
from src.flow_utils import warp_tensor
import torch
import torchvision
import gc
from tqdm import tqdm
from scipy.ndimage import zoom
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from src.utils import *
import sys
sys.path.append("./src/ebsynth/deps/gmflow/")
from gmflow.geometry import flow_warp
import os

# bchw
# 该函数用于展示中间步骤，使用请关闭
def tensor_to_image_from_list(tensor_list, filename, m1, m2):
    illustrate = False
    if not illustrate:
        return
    if not os.path.exists("./illustration"):
        os.makedirs("./illustration")
    tensor_list = F.interpolate(tensor_list, scale_factor=5, mode='nearest')
    for i, tensor in enumerate(tensor_list):
        filepath = f"./illustration/{filename}_{i}.png"
        tensor_to_image(tensor, filepath, m1, m2)
#chw
def tensor_to_image(tensor, output_path, m1, m2):
    tensor = tensor.cpu()
    tensor_min = m1
    tensor_max = m2
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    tensor = tensor * 255
    tensor = tensor.to(torch.uint8)
    tensor = tensor.permute(1, 2, 0)
    h,w,c = tensor.shape
    #print(output_path,":",h,w,c)
    if c > 3:
        tensor = tensor[:,:,:3]
    if c < 3:
        fixtensor = torch.zeros(h,w,1)
        tensor = torch.cat((fixtensor, tensor), dim=2)
        print(tensor.shape)
    image = Image.fromarray(np.uint8(tensor.numpy()))
    image.save(output_path)

def upsample_noise(X, N):
    b, c, h, w = X.shape
    Z = torch.randn(b, c, N*h, N*w).to(torch.float32)
    Z = Z.cuda()
    Z_mean = Z.unfold(2, N, N).unfold(3, N, N).mean((4, 5))
    Z_mean = F.interpolate(Z_mean, scale_factor=N, mode='nearest')
    X = F.interpolate(X, scale_factor=N, mode='nearest')
    return X / N + Z - Z_mean

# old implement 已废弃
@torch.no_grad()
def warp_tensor_noise(sample, flows_centralized, occs, saliency, backyard, N):
    b, c, h, w = sample.shape
    h = int(h / N)
    w = int(w / N)
    sc = int(N*h) * 1.0 / flows_centralized[0].shape[2]
    bwd_flow_ = F.interpolate(flows_centralized[1] * sc, size=(int(N*h), int(N*w)), mode='bilinear')
    fwd_flow_ = F.interpolate(flows_centralized[0] * sc, size=(int(N*h), int(N*w)), mode='bilinear')
    result = torch.randn(b, c, h, w).to(torch.float32).cuda()
    bwd = torch.zeros(b-1, 2, N*h, N*w)
    fwd = torch.zeros(b-1, 2, N*h, N*w)
    #tensor_to_image_from_list(bwd_flow_,"bwd", -50, 50)
    #tensor_to_image_from_list(fwd_flow_,"fwd", -50, 50)
    for bi in tqdm(range(b)):
        for ci in range(c):
            for hi in range(h):
                for wi in range(w):
                    count = 0
                    summ = 0
                    already_calculated = []
                    for ii in range(N):
                        for jj in range(N):
                            backx = int(hi*N+ii)
                            backy = int(wi*N+jj)
                            newbackx = int(fwd_flow_[bi][1][backx][backy])
                            newbacky = int(fwd_flow_[bi][0][backx][backy])
                            backx += newbackx
                            backy += newbacky
                            if backx < 0 or backx >= h*N or backy < 0 or backy >= w*N:
                                continue
                            #print("backx=",backx,",backy=",backy)
                            if (backx,backy) in already_calculated:
                                continue
                            already_calculated.append((backx,backy))
                            count += 1
                            summ += sample[bi][ci][backx][backy]
                    if count == 0:
                        continue
                    result[bi][ci][hi][wi] = summ / (count ** 0.5)
    return result.to(sample.dtype)

# new implement: faster  已废弃
@torch.no_grad()
def better_warp_tensor_noise(sample, flows_centralized, occs, saliency, backyard, N):
    b, c, h, w = sample.shape
    h = int(h / N)
    w = int(w / N)
    sc = int(N*h) * 1.0 / flows_centralized[0].shape[2]
    bwd_flow_ = F.interpolate(flows_centralized[1] * sc, size=(int(N*h), int(N*w)), mode='bilinear')
    fwd_flow_ = F.interpolate(flows_centralized[0] * sc, size=(int(N*h), int(N*w)), mode='bilinear')
    
    # Initialize result tensor directly on GPU
    result = torch.randn(b, c, h, w, device='cuda', dtype=torch.float32)
    
    # Use meshgrid to create indices for the N x N grid
    grid_x, grid_y = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    grid_x = grid_x.flatten().to(sample.device)
    grid_y = grid_y.flatten().to(sample.device)
    
    for bi in tqdm(range(b)):
        for ci in range(c):
            for hi in range(h):
                for wi in range(w):
                    backx = hi * N + grid_x
                    backy = wi * N + grid_y
                    newbackx = (backx + fwd_flow_[bi, 1, backx, backy]).long()
                    newbacky = (backy + fwd_flow_[bi, 0, backx, backy]).long()
                    
                    # Mask to filter valid indices
                    valid_mask = (newbackx >= 0) & (newbackx < N*h) & (newbacky >= 0) & (newbacky < N*w)
                    
                    if valid_mask.sum() == 0:
                        continue
                    
                    valid_backx = newbackx[valid_mask]
                    valid_backy = newbacky[valid_mask]
                    
                    # Gather the values from sample
                    gathered_values = sample[bi, ci, valid_backx, valid_backy]
                    
                    # Compute the mean of the gathered values
                    result[bi, ci, hi, wi] = gathered_values.sum() / (valid_mask.sum() ** 0.5)
    
    return result.to(sample.dtype)

# new new implement: fastest
@torch.no_grad()
def best_warp_tensor_noise(sample, flows_centralized, occs, saliency, backyard, N):
    b, c, h, w = sample.shape
    h = h // N
    w = w // N
    sc = (N * h) * 1.0 / flows_centralized[0].shape[2]
    bwd_flow_ = F.interpolate(flows_centralized[1] * sc, size=(N * h, N * w), mode='bilinear')
    fwd_flow_ = F.interpolate(flows_centralized[0] * sc, size=(N * h, N * w), mode='bilinear')

    # Initialize result tensor directly on GPU
    result = torch.zeros(b, c, h, w, device=sample.device, dtype=torch.float32)
    
    # Use meshgrid to create indices for the N x N grid
    grid_x, grid_y = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    grid_x = grid_x.flatten().to(sample.device)
    grid_y = grid_y.flatten().to(sample.device)
    #print(grid_x, grid_y)
    
    # Create the full grid of (h, w) indices
    hi_indices = torch.arange(h, device=sample.device).repeat_interleave(w)
    wi_indices = torch.arange(w, device=sample.device).repeat(h)
    #print(hi_indices, wi_indices)

    # for bi in tqdm(range(b)):
    for bi in range(b):
        for ci in range(c):
            # Initialize count and summation tensors
            count = torch.zeros(h * w, device=sample.device)
            summ = torch.zeros(h * w, device=sample.device)

            # Create backx and backy for all N x N offsets
            backx = (hi_indices[:, None] * N + grid_x).flatten()
            backy = (wi_indices[:, None] * N + grid_y).flatten()
            #print(backx, backy)

            # Calculate new positions using forward flow
            newbackx = (backx + fwd_flow_[bi, 1, backx, backy]).long()
            newbacky = (backy + fwd_flow_[bi, 0, backx, backy]).long()

            # Mask to filter valid indices
            valid_mask = (newbackx >= 0) & (newbackx < N * h) & (newbacky >= 0) & (newbacky < N * w)
            #print(valid_mask)
            valid_backx = newbackx[valid_mask]
            valid_backy = newbacky[valid_mask]
            #print(valid_backx, valid_backy)

            # Create 1D indices for summation
            indices = (backx // N * w + backy // N)
            indices = indices[valid_mask]

            # Gather the values from sample
            gathered_values = sample[bi, ci, valid_backx, valid_backy]

            # Sum and count valid gathered values
            summ.index_add_(0, indices, gathered_values)
            count.index_add_(0, indices, torch.ones_like(gathered_values, device=sample.device))

            # Reshape count and summ to (h, w)
            count = count.view(h, w)
            summ = summ.view(h, w)

            # Avoid division by zero
            mask = count > 0
            result[bi, ci][mask] = (summ[mask] / (count[mask] ** 0.5))
            result[bi, ci][~mask] = torch.randn_like(result[bi, ci][~mask])

    return result.to(sample.dtype)

# 请调用该函数
# pure_noise: 初始的高斯分布噪声，最后的结果与之大小一致
# flows_centralized：从所有帧到中间帧的光流
# 后三个参数没用到
def warp_pure_noise(pure_noise, flows_centralized, occs, saliency, unet_chunk_size=1):
    #bchw
    N = 2
    B = pure_noise.shape[0]
    b, c, h, w = pure_noise.shape
    backyard = torch.randn(b, c, N*h, N*w).to(torch.float32).cuda()
    pure_noise = pure_noise[0:1].repeat(B,1,1,1).detach()
    pure_noise = upsample_noise(pure_noise, N)
    pure_noise = pure_noise[0:1].repeat(B,1,1,1).detach()
    warped_noise = best_warp_tensor_noise(pure_noise, flows_centralized, occs, saliency, backyard, N)
    tensor_to_image_from_list(warped_noise,"purenoiseswarped1", -2, 2)
    return warped_noise.to(pure_noise.dtype)
