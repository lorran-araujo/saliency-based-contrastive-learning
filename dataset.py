import torch
import os
import random
from pathlib import Path
import numpy as np
from skimage import io
from utils import convert_from_color, get_random_pos, load_icmbio_label, resize_label_mask
import matplotlib.pyplot as plt

EXTS = ['tif', 'tiff', 'jpg', 'png']

class DatasetIcmbio(torch.utils.data.Dataset):
    def __init__(self, data_files, window_size, len, label_files=None, n_channels=3, cache=True,
                 augmentation=True, icmbio_labels=False, original_labels_dir=None):
        super(DatasetIcmbio, self).__init__()
        
        self.data_files = data_files
        self.label_files = label_files
        self.window_size = window_size
        self.n_channels = n_channels
        self.cache = cache
        self.augmentation = augmentation
        self.len = len
        self.icmbio_labels = icmbio_labels
        self.original_labels_dir = original_labels_dir

        for f in self.data_files:
            if not os.path.isfile(f):
                raise KeyError(f'{f} is not a file!')

        if self.label_files:
            for f in self.label_files:
                if not os.path.isfile(f):
                    raise KeyError(f'{f} is not a file!')

        self.data_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        return self.len

    def get_dataset(self):
        return self.data_files, self.label_files

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
        
        return tuple(results)

    def __getitem__(self, i):
        random_idx = random.randint(0, len(self.data_files) - 1)

        if random_idx in self.data_cache_:
            data = self.data_cache_[random_idx]
        else:
            data = io.imread(self.data_files[random_idx]).transpose((2,0,1))
            if data.ndim == 2:
                data = np.stack([data]*3, axis=0)
            elif data.ndim == 3 and data.shape[0] == 4:
                data = data[:3, :, :]
            data = 1/255 * np.asarray(data, dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        x1, x2, y1, y2 = get_random_pos(data, self.window_size)
        data_p = data[:, x1:x2, y1:y2]

        if self.label_files:
            if random_idx in self.label_cache_:
                label = self.label_cache_[random_idx]
            else:
                label_file = self.label_files[random_idx]
                if self.icmbio_labels:
                    original_file = (
                        Path(self.original_labels_dir) / Path(label_file).name
                        if self.original_labels_dir is not None else None
                    )
                    if original_file is not None and original_file.is_file():
                        label = resize_label_mask(load_icmbio_label(original_file), (data.shape[-1], data.shape[-2]))
                    else:
                        label = load_icmbio_label(label_file)
                        if label.shape != data.shape[-2:]:
                            raise ValueError(f'Label {label_file} has shape {label.shape}, expected {data.shape[-2:]}.')
                    label = label.astype('int64')
                else:
                    label = np.asarray(convert_from_color(io.imread(label_file)), dtype='int64')
                if self.cache:
                    self.label_cache_[random_idx] = label
            label_p = label[x1:x2, y1:y2]

            if self.augmentation:
                data_p, label_p = self.data_augmentation(data_p, label_p)
            return data_p, label_p
        else:
            if self.augmentation:
                data_p, = self.data_augmentation(data_p)
            return data_p

    def _visualize(self, image, mask=None, original_image=None, original_mask=None):
        fontsize = 18

        if mask is None:
            plt.imshow(image.transpose(1, 2, 0))
            plt.title("Image (no mask)", fontsize=fontsize)
            plt.show()
            return

        if original_image is None and original_mask is None:
            f, ax = plt.subplots(2, 1, figsize=(8, 8))
            ax[0].imshow(image)
            ax[1].imshow(mask)
        else:
            fig, ax = plt.subplots(2, 2, figsize=(8, 8))

            ax[0, 0].imshow(original_image)
            ax[0, 0].set_title('Original image', fontsize=fontsize)
            
            ax[1, 0].imshow(original_mask)
            ax[1, 0].set_title('Original mask', fontsize=fontsize)
            
            ax[0, 1].imshow(image.detach().cpu().numpy().transpose(1,2,0))
            ax[0, 1].set_title('Transformed image', fontsize=fontsize)
            
            ax[1, 1].imshow(mask)
            ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

            fig.savefig(f"./tmp/dataset", dpi=fig.dpi, bbox_inches='tight')
            plt.close(fig)
