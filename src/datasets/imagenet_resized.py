import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path
import zipfile
import os
from itertools import chain
import io


def _normpath(path):
  path = os.path.normpath(path)
  if (path.startswith('.')
      or os.path.isabs(path)
      or path.endswith('~')
      or os.path.basename(path).startswith('.')):
    return None
  return path

def iter_zip(arch_f):
    """Iterate over zip archive."""
    with open(arch_f, 'rb') as fobj:
        z = zipfile.ZipFile(fobj)
        for member in z.infolist():
            extract_file = z.open(member)
            if member.is_dir():  # Filter directories  # pytype: disable=attribute-error
                continue
            path = _normpath(member.filename)
            if not path:
                continue
            yield [path, extract_file]


class ImagenetResized(data.Dataset):
    
    def __init__(self, image_size, image_set, 
        root="/mnt/SAMSUNG/datasets/imagenet/", 
        transform=None, transform_target=None, normalise=True):

        self.size = image_size
        if self.size in {16,32}:
            data_paths = [
                Path(root) / f"imagenet{image_size}/Imagenet{image_size}_{image_set}_npz.zip"
            ]
        elif self.size == 64:
            if image_set == "train":
                data_paths = [
                    Path(root) / f"imagenet{image_size}/Imagenet{image_size}_{image_set}_part1_npz.zip",
                    Path(root) / f"imagenet{image_size}/Imagenet{image_size}_{image_set}_part2_npz.zip"
                ]
            else:
                data_paths = [
                    Path(root) / f"imagenet{image_size}/Imagenet{image_size}_{image_set}_npz.zip"
                ]
        else:
            raise NameError("specified size not supported")

        self.data_paths = data_paths

        self.transform = transform
        self.transform_target = transform_target
        self.normalise = normalise

        self.data = []
        self.labels = []

        for name, archive_file in chain(*[iter_zip(f) for f in self.data_paths]):
            content = archive_file.read()
            if content:
                fobj_mem = io.BytesIO(content)
                data = np.load(fobj_mem, allow_pickle=False)
                size = self.size
                self.data.append(data['data'])
                self.labels.append(data['labels'] -1)
                # for i, (image, label) in enumerate(zip(data['data'], data['labels'])):
                #     image = image.reshape(3, size, size).transpose(1, 2, 0)
                #     self.data.append(image)
                #     self.labels.append(label - 1)
        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)

    def __getitem__(self, index):            
        x = torch.from_numpy(self.data[index].reshape(3,self.size, self.size).transpose(1,2,0)).float()
        y = torch.from_numpy(self.labels[index:index+1]).long().squeeze()

        if self.transform:
            x = self.transform(x)
        if self.normalise:
            x = x / 255.

        if self.transform_target:
            y = self.transform_target(y)
        return x, y

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


            
    # def __getitem__(self, index):            
    #     x = torch.from_numpy(self.data[index,:,:,:]).float()
    #     y = torch.from_numpy(self.target[index,:,:,:]).long().squeeze()

    #     if self.transform:
    #         x = self.transform(x)
    #     if self.normalise:
    #         x = x / 255.

    #     if self.transform_target:
    #         y = self.transform_target(y)
    #     return x, y

# def __len__(self):
#     return self.data.shape[0]

    # def __iter__(self):
    #     for name, archive_file in chain(*[iter_zip(f) for f in self.data_paths]):
    #         content = archive_file.read()
    #         if content:
    #             fobj_mem = io.BytesIO(content)
    #             data = np.load(fobj_mem, allow_pickle=False)
    #             size = self.size
    #             for i, (image, label) in enumerate(zip(data['data'], data['labels'])):
    #                 # The data is packed flat as CHW. It is converted to HWC
    #                 x = image.reshape(3, size, size).transpose(1, 2, 0)
    #                 x = torch.from_numpy(x).float()
    #                 # Labels are 1 indexed, so 1 is subtracted
    #                 y = torch.tensor(label - 1).long()

    #                 if self.transform:
    #                     x = self.transform(x)
    #                 if self.normalise:
    #                     x = x / 255.

    #                 if self.transform_target:
    #                     y = self.transform_target(y)
                        
    #                 yield x, y