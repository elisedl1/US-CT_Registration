import torch
import numpy as np
import SimpleITK as sitk
from utils.helpers import sitk_euler_to_matrix

class TransformManager:
    
    def __init__(self, flat_params, center, device='cuda', inverse=False):

        # build SimpleITK transform once
        tx = sitk.Euler3DTransform()
        tx.SetCenter(center.tolist() if torch.is_tensor(center) else center)
        tx.SetParameters(flat_params.tolist() if torch.is_tensor(flat_params) else flat_params)
        
        if inverse:
            tx = tx.GetInverse()
        
        M = sitk_euler_to_matrix(tx)
        self.M = torch.from_numpy(M).to(device=device, dtype=torch.float64)
        self.device = device
        
        self.R = self.M[:3, :3]
    
    def apply(self, pts):
        
        # transform points using homogeneous coordinates
        pts = pts.to(dtype=self.M.dtype)
        
        N = pts.shape[0]
        pts_h = torch.cat([pts, torch.ones((N, 1), device=pts.device, dtype=pts.dtype)], dim=1)
        return (pts_h @ self.M.T)[:, :3]
    
    def apply_to_axes(self, axes):
        # rotate direction vectors
        if axes.ndim == 1:
            return self.R @ axes
        else:
            return (axes @ self.R.T)