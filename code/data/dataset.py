import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LabColorizationDataset(Dataset):
    def __init__(self, root_dir1, root_dir2):
        """
        Args:
            root_dir1 (string): Directory containing 'gray_scale.npy'.
            root_dir2 (string): Directory containing multiple 'abX.npy' files.
        """
        self.L_path = os.path.join(root_dir1, "gray_scale.npy")
        self.ab_dir = os.path.join(root_dir2, "ab")

        # Load grayscale images
        print("start loading gray scale")
        self.L_data = np.load(self.L_path).astype(np.float32)  # (N, H, W)
        self.L_data = np.expand_dims(self.L_data, axis=1)  # (N, 1, H, W)  <-- FIXED!
        print("finish loading gray scale")

        # Load and concatenate all ab .npy files
        print("start loading ab")
        
        ab_files = sorted([f for f in os.listdir(self.ab_dir) if f.startswith("ab") and f.endswith(".npy")])
        ab_arrays = [np.load(os.path.join(self.ab_dir, f)).astype(np.float32) for f in ab_files]
        self.ab_data = np.concatenate(ab_arrays, axis=0)  # (N, 2, H, W)
        print(len(self.ab_data), len(self.L_data))
        print("finish loading ab")

        # Ensure data consistency
        assert self.L_data.shape[0] == self.ab_data.shape[0], \
            f"Mismatch between grayscale ({self.L_data.shape[0]}) and ab ({self.ab_data.shape[0]}) data!"

    def __len__(self):
        return self.L_data.shape[0]

    def __getitem__(self, idx):
        # Extract L and ab tensors
        L = self.L_data[idx] / 255.0  # Normalize to [0, 1]
        ab = (self.ab_data[idx] / 128.0) - 1  # Normalize to [-1, 1]

        # Convert to PyTorch tensors
        L = torch.tensor(L, dtype=torch.float32)
        ab = torch.tensor(ab, dtype=torch.float32)

        return L, ab
