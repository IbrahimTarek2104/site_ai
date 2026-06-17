import numpy as np
import rasterio
import torch
import cv2

class ProductionInferencePipeline:
    def __init__(self, patch_size=64):
        self.patch_size = patch_size

    def extract_and_normalize(self, coverage_path, population_path, elevation_path):
        """Loads and builds a clean 3-band input stack directly from uploaded file pointers."""
        with rasterio.open(coverage_path) as src_cov:
            cov = src_cov.read(1).astype(np.float32)
            meta = src_cov.profile
            
        with rasterio.open(population_path) as src_pop:
            pop = src_pop.read(1).astype(np.float32)
            
        with rasterio.open(elevation_path) as src_elev:
            elev = src_elev.read(1).astype(np.float32)

        # Apply deterministic data cleaning rules
        cov = np.clip(cov, 0, 1)
        pop = np.maximum(pop, 0.0)
        pop[pop < 0] = 0.0
        elev[elev < -500] = 0.0

        # Run safe Min-Max normalization transformations
        norm_cov = cov
        norm_pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-8)
        norm_elev = (elev - elev.min()) / (elev.max() - elev.min() + 1e-8)

        # Combine into a clean 3-band processing stack
        features_stack = np.dstack((norm_cov, norm_pop, norm_elev))
        return features_stack, meta

    def generate_gaussian_patches(self, stack):
        """Slices variable-sized rasters into patches with clean border boundaries."""
        h, w, c = stack.shape
        p = self.patch_size
        
        n_h = int(np.ceil(h / p))
        n_w = int(np.ceil(w / p))
        
        pad_bottom = (n_h * p) - h
        pad_right = (n_w * p) - w
        
        padded_stack = cv2.copyMakeBorder(
            stack, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0
        )
        
        patches = []
        coords = []
        for i in range(n_h):
            for j in range(n_w):
                patches.append(padded_stack[i*p:(i+1)*p, j*p:(j+1)*p, :])
                coords.append((i, j))
                
        return patches, coords, (h, w, n_h * p, n_w * p)