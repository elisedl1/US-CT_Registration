from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import json
import SimpleITK as sitk 


class PyNrrdParser:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.image = sitk.ReadImage(file_path)
        self.array = sitk.GetArrayFromImage(self.image).astype(np.float32)

        self.sitk_spacing = np.array(self.image.GetSpacing(), dtype=np.float64)
        self.sitk_origin = np.array(self.image.GetOrigin(), dtype=np.float64)
        self.sitk_direction = np.array(self.image.GetDirection(), dtype=np.float64).reshape(3, 3)

        self.size = np.array(self.array.shape, dtype=np.int64)
        self.spacing_zxy = self.sitk_spacing[::-1].astype(np.float64)
        self.origin_zxy = self.sitk_origin[::-1].astype(np.float64)
        self.end_zxy = self.origin_zxy + (self.size - 1) * self.spacing_zxy
        
        # GPU CASHING
        self.array_gpu = None  
        self.transform_cache = {}  


    def compute_positions(self, indices: torch.Tensor) -> np.ndarray:
        idx_np = indices.cpu().numpy().astype(np.float64)
        pts = []
        for idx in idx_np:
            itk_index = (float(idx[2]), float(idx[1]), float(idx[0]))
            pt = self.image.TransformContinuousIndexToPhysicalPoint(itk_index)
            pts.append(pt)
        return np.array(pts, dtype=np.float64)


    def position_to_index(self, positions: np.ndarray) -> np.ndarray:

        positions = np.asarray(positions, dtype=np.float64)  
        
        # invert the 3x3 direction matrix
        dir_inv = np.linalg.inv(self.sitk_direction) 
        offset = positions - self.sitk_origin 

        # apply inverse direction to get continuous index in x,y,z order
        idx_xyz = offset @ dir_inv.T / self.sitk_spacing 

        idx_zxy = idx_xyz[:, [2, 1, 0]]

        return idx_zxy


    @staticmethod
    def _trilinear_interpolate(array: np.ndarray, coords_zxy: np.ndarray) -> np.ndarray:
        """
        Vectorized trilinear interpolation for N points.
        coords_zxy: (N, 3) array of continuous indices
        Returns: (N,) interpolated values
        """
        D, H, W = array.shape
        coords = np.asarray(coords_zxy, dtype=np.float64)
        z, y, x = coords[:,0], coords[:,1], coords[:,2]

        # corner indices
        x0 = np.clip(np.floor(x).astype(int), 0, W-2)
        y0 = np.clip(np.floor(y).astype(int), 0, H-2)
        z0 = np.clip(np.floor(z).astype(int), 0, D-2)

        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        xd = x - x0
        yd = y - y0
        zd = z - z0

        # gather 8 corner values
        c000 = array[z0, y0, x0]
        c001 = array[z0, y0, x1]
        c010 = array[z0, y1, x0]
        c011 = array[z0, y1, x1]
        c100 = array[z1, y0, x0]
        c101 = array[z1, y0, x1]
        c110 = array[z1, y1, x0]
        c111 = array[z1, y1, x1]

        # vectorized interpolation
        c00 = c000*(1 - xd) + c001*xd
        c01 = c010*(1 - xd) + c011*xd
        c10 = c100*(1 - xd) + c101*xd
        c11 = c110*(1 - xd) + c111*xd

        c0 = c00*(1 - yd) + c01*yd
        c1 = c10*(1 - yd) + c11*yd

        return c0*(1 - zd) + c1*zd

    
    def sample_at_physical_points(self, positions: np.ndarray) -> np.ndarray:
        cont_idx_zxy = self.position_to_index(positions)
        return self._trilinear_interpolate(self.array, cont_idx_zxy)

    def get_bspline_grid(self, node_spacing_mm: float) -> Tuple[int, int, int]:
        physical_lengths = (self.size - 1).astype(np.float64) * self.spacing_zxy
        counts = np.floor(physical_lengths / float(node_spacing_mm)).astype(np.int64)
        counts = np.maximum(counts, 1)
        return tuple(counts.tolist())

    def sample_at_physical_points_gpu(self, positions: torch.Tensor) -> torch.Tensor:

        device = positions.device
        dtype = positions.dtype
        
        # OPTIMIZATION 1: cache GPU array (convert once, not every call)
        if self.array_gpu is None or self.array_gpu.device != device:
            self.array_gpu = torch.from_numpy(self.array).to(device=device, dtype=torch.float32)
        
        array_t = self.array_gpu
        if array_t.dtype != dtype:
            array_t = array_t.to(dtype)
        
        # OPTIMIZATION 2: cache transform parameters per device/dtype
        cache_key = (str(device), str(dtype))
        if cache_key not in self.transform_cache:
            origin = torch.tensor(self.sitk_origin, device=device, dtype=dtype)
            spacing = torch.tensor(self.sitk_spacing, device=device, dtype=dtype)
            dir_mat = torch.tensor(self.sitk_direction, device=device, dtype=dtype)
            dir_inv = torch.linalg.inv(dir_mat)
            
            self.transform_cache[cache_key] = {
                'origin': origin,
                'spacing': spacing,
                'dir_inv': dir_inv
            }
        
        cache = self.transform_cache[cache_key]
        origin = cache['origin']
        spacing = cache['spacing']
        dir_inv = cache['dir_inv']
        
        # transform physical positions to voxel indices
        offset = positions - origin
        idx_xyz = (offset @ dir_inv.T) / spacing
        idx_zxy = idx_xyz[:, [2, 1, 0]]
        
        return self._trilinear_interpolate_gpu(array_t, idx_zxy)
    
    
    @staticmethod
    def _trilinear_interpolate_gpu(array_t: torch.Tensor, coords_zxy: torch.Tensor) -> torch.Tensor:

        D, H, W = array_t.shape
        z = coords_zxy[:, 0]
        y = coords_zxy[:, 1]
        x = coords_zxy[:, 2]

        # floor / ceil
        z0 = torch.clamp(z.floor().long(), 0, D - 2)
        y0 = torch.clamp(y.floor().long(), 0, H - 2)
        x0 = torch.clamp(x.floor().long(), 0, W - 2)

        z1 = z0 + 1
        y1 = y0 + 1
        x1 = x0 + 1

        # interpolation weights
        zd = z - z0.float()
        yd = y - y0.float()
        xd = x - x0.float()

        # gather 8 corners
        c000 = array_t[z0, y0, x0]
        c001 = array_t[z0, y0, x1]
        c010 = array_t[z0, y1, x0]
        c011 = array_t[z0, y1, x1]
        c100 = array_t[z1, y0, x0]
        c101 = array_t[z1, y0, x1]
        c110 = array_t[z1, y1, x0]
        c111 = array_t[z1, y1, x1]

        # interpolate along x
        c00 = c000 * (1 - xd) + c001 * xd
        c01 = c010 * (1 - xd) + c011 * xd
        c10 = c100 * (1 - xd) + c101 * xd
        c11 = c110 * (1 - xd) + c111 * xd

        # interpolate along y
        c0 = c00 * (1 - yd) + c01 * yd
        c1 = c10 * (1 - yd) + c11 * yd

        # interpolate along z
        return c0 * (1 - zd) + c1 * zd


    def get_tensor(self, scale_mm: float = 1.0, remove_background: bool = False, device='cpu') -> torch.Tensor:
        arr = self.array.copy()
        if remove_background:
            arr[arr==0] = 0.0
        sigma_vox = (scale_mm / self.spacing_zxy).astype(np.float64)
        arr = gaussian_filter(arr, sigma=sigma_vox)
        return torch.from_numpy(arr.astype(np.float32)).to(device)









        


class TagFileParser:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        self.src_landmarks, self.trg_landmarks = self.extract_landmarks()

    def extract_landmarks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        src_landmarks_list = []
        trg_landmarks_list = []
        with open(self.file_name) as f:
            data = f.readlines()
        stripped_data = [item.strip() for item in data]
        for index, line in enumerate(stripped_data):
            if line.startswith('Points'):
                break
        landmarks = stripped_data[index + 1:]
        for landmark in landmarks:
            srcx, srcy, srcz, trgx, trgy, trgz, _ = landmark.split()
            src_landmarks_list.append([float(srcx), float(srcy), float(srcz)])
            trg_landmarks_list.append([float(trgx), float(trgy), float(trgz)])
        src_landmarks = torch.tensor(src_landmarks_list)
        trg_landmarks = torch.tensor(trg_landmarks_list)
        return src_landmarks, trg_landmarks


class SlicerJsonTagParser:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        self.landmarks = self.extract_landmarks()

    def extract_landmarks(self) -> torch.Tensor:
        with open(self.file_name, 'r') as f:
            data = json.load(f)
        
        markups = data.get("markups", [])
        if len(markups) == 0:
            raise ValueError(f"No markups found in {self.file_name}")
        
        # collection all control points from markup
        control_points = markups[0].get("controlPoints", [])
        if not control_points:
            raise ValueError(f"No control points found in markup of {self.file_name}")
        
        points = [cp["position"] for cp in control_points]

        # homogenous
        # points_h = np.hstack([points, np.ones((points.shape[0], 1))])

        return torch.tensor(points, dtype=torch.float32)