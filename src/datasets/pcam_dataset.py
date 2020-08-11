import h5py
import torch
import torch.utils.data as data
from pathlib import Path

class PatchCamelyonDataset(data.Dataset):
    
    def __init__(self, root, image_set, transform=None, transform_target=None,
                 normalise=True):
        data_path = Path(root) / f"camelyonpatch_level_2_split_{image_set}_x.h5"
        target_path = Path(root) / f"camelyonpatch_level_2_split_{image_set}_y.h5"
        self.data = h5py.File(data_path)["x"]
        self.target = h5py.File(target_path)["y"]

        self.transform = transform
        self.transform_target = transform_target
        self.normalise = normalise
    
    def __getitem__(self, index):            
        x = torch.from_numpy(self.data[index,:,:,:]).float()
        y = torch.from_numpy(self.target[index,:,:,:]).long().squeeze()

        if self.transform:
            x = self.transform(x)
        if self.normalise:
            x = x / 255.

        if self.transform_target:
            y = self.transform_target(y)
        return x, y

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]