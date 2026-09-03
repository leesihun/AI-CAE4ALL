import torch
import numpy as np
from sklearn.decomposition import PCA
import pickle
import os
import math

class PCAPreprocessor:
    """PCA preprocessing for image inputs to reduce dimensionality while preserving spatial structure"""
    
    def __init__(self, n_components=1024, patch_size=None, save_dir='model_save'):
        """
        Initialize PCA preprocessor
        
        Args:
            n_components (int): Number of PCA components to keep
            patch_size (int, optional): If specified, apply PCA to patches instead of full images
            save_dir (str): Directory to save/load PCA models
        """
        self.n_components = n_components
        self.patch_size = patch_size
        self.save_dir = save_dir
        self.pca = None
        self.is_fitted = False
        self.original_shape = None
        self.output_shape = None
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def _get_pca_filename(self):
        """Generate filename for PCA model"""
        if self.patch_size:
            return f'pca_patch{self.patch_size}_comp{self.n_components}.pkl'
        else:
            return f'pca_full_comp{self.n_components}.pkl'
    
    def fit(self, images):
        """
        Fit PCA on training images
        
        Args:
            images (np.ndarray): Training images of shape (n_samples, height*width) or (n_samples, height, width)
        """
        if len(images.shape) == 3:
            # Convert (n_samples, height, width) to (n_samples, height*width)
            n_samples, height, width = images.shape
            self.original_shape = (height, width)
            images_flat = images.reshape(n_samples, -1)
        else:
            # Already flattened
            images_flat = images
            # Assume square images
            side_length = int(math.sqrt(images_flat.shape[1]))
            self.original_shape = (side_length, side_length)
        
        print(f"Fitting PCA on {images_flat.shape[0]} images of size {self.original_shape}")
        
        if self.patch_size:
            # Patch-based PCA (more complex, preserves spatial structure better)
            self._fit_patch_pca(images_flat)
        else:
            # Full image PCA
            self._fit_full_pca(images_flat)
        
        # Save fitted PCA
        self.save()
        
    def _fit_full_pca(self, images_flat):
        """Fit PCA on full flattened images"""
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(images_flat)
        
        # Calculate output shape (square arrangement of components)
        out_side = int(math.sqrt(self.n_components))
        if out_side * out_side != self.n_components:
            # Adjust to nearest square
            out_side = int(math.ceil(math.sqrt(self.n_components)))
            self.n_components = out_side * out_side
            print(f"Adjusted n_components to {self.n_components} for square output")
            
        self.output_shape = (out_side, out_side)
        self.is_fitted = True
        
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA fitted: {self.n_components} components explain {explained_var:.1%} of variance")
        
    def _fit_patch_pca(self, images_flat):
        """Fit PCA on image patches (preserves spatial structure)"""
        height, width = self.original_shape
        
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(f"Image size {self.original_shape} not divisible by patch size {self.patch_size}")
        
        # Extract all patches from all images
        n_samples = images_flat.shape[0]
        images_2d = images_flat.reshape(n_samples, height, width)
        
        patches_per_dim = height // self.patch_size
        n_patches_per_image = patches_per_dim * patches_per_dim
        patch_dim = self.patch_size * self.patch_size
        
        # Collect all patches
        all_patches = np.zeros((n_samples * n_patches_per_image, patch_dim))
        
        idx = 0
        for img in images_2d:
            for i in range(patches_per_dim):
                for j in range(patches_per_dim):
                    patch = img[i*self.patch_size:(i+1)*self.patch_size, 
                               j*self.patch_size:(j+1)*self.patch_size]
                    all_patches[idx] = patch.flatten()
                    idx += 1
        
        # Fit PCA on all patches
        components_per_patch = self.n_components // n_patches_per_image
        if components_per_patch < 1:
            components_per_patch = 1
            self.n_components = n_patches_per_image
            
        self.pca = PCA(n_components=components_per_patch)
        self.pca.fit(all_patches)
        
        # Output shape is arranged as a grid of patch features
        self.output_shape = (patches_per_dim, patches_per_dim, components_per_patch)
        self.is_fitted = True
        
        explained_var = np.sum(self.pca.explained_variance_ratio_)  
        print(f"Patch PCA fitted: {components_per_patch} components per patch, {explained_var:.1%} variance explained")
        
    def transform(self, images):
        """
        Transform images using fitted PCA
        
        Args:
            images: Images to transform, shape (n_samples, height*width) or (n_samples, height, width)
            
        Returns:
            torch.Tensor: PCA-transformed images ready for CNN input
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first or load() a pre-fitted model.")
            
        if len(images.shape) == 3:
            images_flat = images.reshape(images.shape[0], -1)
        else:
            images_flat = images
            
        if self.patch_size:
            return self._transform_patch_pca(images_flat)
        else:
            return self._transform_full_pca(images_flat)
    
    def _transform_full_pca(self, images_flat):
        """Transform using full image PCA"""
        pca_coeffs = self.pca.transform(images_flat)  # (n_samples, n_components)
        
        # Reshape to 2D for CNN input
        n_samples = pca_coeffs.shape[0]
        pca_images = pca_coeffs.reshape(n_samples, 1, self.output_shape[0], self.output_shape[1])
        
        return torch.tensor(pca_images, dtype=torch.float32)
    
    def _transform_patch_pca(self, images_flat):
        """Transform using patch-based PCA"""
        height, width = self.original_shape
        n_samples = images_flat.shape[0]
        images_2d = images_flat.reshape(n_samples, height, width)
        
        patches_per_dim = height // self.patch_size
        components_per_patch = self.output_shape[2]
        
        # Transform each patch
        pca_features = np.zeros((n_samples, patches_per_dim, patches_per_dim, components_per_patch))
        
        for sample_idx, img in enumerate(images_2d):
            for i in range(patches_per_dim):
                for j in range(patches_per_dim):
                    patch = img[i*self.patch_size:(i+1)*self.patch_size, 
                               j*self.patch_size:(j+1)*self.patch_size]
                    patch_flat = patch.flatten().reshape(1, -1)
                    patch_pca = self.pca.transform(patch_flat)[0]
                    pca_features[sample_idx, i, j, :] = patch_pca
        
        # Reshape for CNN: (n_samples, channels, height, width)
        pca_images = pca_features.transpose(0, 3, 1, 2)  # (n_samples, components_per_patch, patches_per_dim, patches_per_dim)
        
        return torch.tensor(pca_images, dtype=torch.float32)
    
    def save(self):
        """Save fitted PCA model"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted PCA model")
            
        filepath = os.path.join(self.save_dir, self._get_pca_filename())
        
        save_dict = {
            'pca': self.pca,
            'n_components': self.n_components,
            'patch_size': self.patch_size,
            'original_shape': self.original_shape,
            'output_shape': self.output_shape,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
            
        print(f"PCA model saved to {filepath}")
    
    def load(self):
        """Load pre-fitted PCA model"""
        filepath = os.path.join(self.save_dir, self._get_pca_filename())
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PCA model not found at {filepath}")
            
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
            
        self.pca = save_dict['pca']
        self.n_components = save_dict['n_components']
        self.patch_size = save_dict['patch_size']
        self.original_shape = save_dict['original_shape']
        self.output_shape = save_dict['output_shape']
        self.is_fitted = save_dict['is_fitted']
        
        print(f"PCA model loaded from {filepath}")
        
    def get_output_shape(self):
        """Get the output shape after PCA transformation"""
        if not self.is_fitted:
            raise ValueError("PCA not fitted")
        return self.output_shape
        
    def get_output_channels(self):
        """Get number of output channels for CNN"""
        if not self.is_fitted:
            raise ValueError("PCA not fitted") 
            
        if self.patch_size:
            return self.output_shape[2]  # components per patch
        else:
            return 1  # single channel