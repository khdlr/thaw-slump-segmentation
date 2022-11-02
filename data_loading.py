# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import xarray
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from lib.utils import get_logger
from lib.utils.data import H5Dataset, Augment, Transformed, Scaling
from collections import namedtuple
from tqdm import tqdm
from pathlib import Path


class DataSources:
    DataSource = namedtuple('DataSource',
            ['name', 'channels', 'normalization_factors'])
    Sentinel2 = DataSource('sentinel2', 13, [10000]*13)
    Planet = DataSource('planet', 4, [3000, 3000, 3000, 3000])
    NDVI = DataSource('ndvi', 1, [20000])
    TCVIS = DataSource('tcvis', 3, [255, 255, 255])
    RelativeElevation = DataSource('relative_elevation', 1, [30000])
    Slope = DataSource('slope', 1, [90])
    LIST = list(sorted([Planet, NDVI, TCVIS, RelativeElevation, Slope, Sentinel2]))
    NAME2SOURCE = {src.name: src for src in LIST}

    def __init__(self, sources):
        sources = (self._get_source(source) for source in sources)
        # Sort Data Sources to be independent of the order given in the config
        self.sources = tuple(sorted(sources))

    def _get_source(self, source):
        if type(source) is str:
            return DataSources.NAME2SOURCE[source]
        elif type(source) is DataSources.DataSource:
            return source
        else:
            raise ValueError(f"Can't convert object {source}"
                             f"of type {type(source)} to DataSource")

    @staticmethod
    def all():
        return DataSources([src.name for src in DataSources.LIST])

    def __iter__(self):
        return self.sources.__iter__()

    def index(self, element):
        return self.sources.index(element)


def make_scaling(data_sources=None):
    if data_sources is None:
        # Use all data sources by default
        data_sources = DataSources.all()
    # Sort Data Sources to be independent of the order given in the config
    data_sources = DataSources(data_sources)

    factors = []
    for source in data_sources:
        factors += source.normalization_factors

    normalize = 1 / torch.tensor(factors, dtype=torch.float32).reshape(-1, 1, 1)
    return Scaling(normalize)


class TimeseriesDataset(Dataset):
  def __init__(self, netcdf_path, tile_size=128, sampling_mode='targets_only'):
    self.data = xarray.open_dataset(netcdf_path, cache=False)
    self.sampling_mode = sampling_mode

    self.tile_size = tile_size
    self.T, self.H, self.W = len(self.data.time), len(self.data.y), len(self.data.x)
    self.H_tile = self.H // self.tile_size
    self.W_tile = self.W // self.tile_size

    if self.sampling_mode == 'targets_only':
      is_ever_positive = (self.data.Mask == 1).any('mask_time').squeeze('mask_band')
      self.positive_points = np.argwhere(is_ever_positive.values)
    elif self.sampling_mode == 'grid_nearest':
      self.T = len(self.data.mask_time)
      self.closest_sample = [np.argmin(np.abs(self.data.time.values - t))
                             for t in self.data.mask_time.values]

  def __getitem__(self, idx):
    if self.sampling_mode == 'grid':
      t, inner_idx = divmod(idx, self.H_tile * self.W_tile)
      y_tile, x_tile = divmod(inner_idx, self.W_tile)
      y0 = y_tile * self.tile_size
      x0 = x_tile * self.tile_size
    elif self.sampling_mode == 'grid_nearest':
      t_mask, inner_idx = divmod(idx, self.H_tile * self.W_tile)
      y_tile, x_tile = divmod(inner_idx, self.W_tile)
      y0 = y_tile * self.tile_size
      x0 = x_tile * self.tile_size
      t = self.closest_sample[t_mask]
    elif self.sampling_mode == 'random':
      t  = int(torch.randint(0, self.T, ()))
      y0 = int(torch.randint(0, self.H - self.tile_size, ()))
      x0 = int(torch.randint(0, self.W - self.tile_size, ()))
    elif self.sampling_mode == 'targets_only':
      t  = int(torch.randint(0, self.T, ()))
      point_idx = int(torch.randint(0, self.positive_points.shape[0], ()))
      contained_y, contained_x = self.positive_points[point_idx]

      y0 = max(0, min(self.H - self.tile_size,
               contained_y - int(torch.randint(0, self.tile_size, ()))))
      x0 = max(0, min(self.W - self.tile_size,
                      contained_x - int(torch.randint(0, self.tile_size, ()))))
    else:
      raise ValueError(f'Unsupported tiling mode: {self.sampling_mode!r}')
    y1 = y0 + self.tile_size
    x1 = x0 + self.tile_size

    tile = self.data.Sentinel2[t, :, y0:y1, x0:x1]
    tile_date = self.data.time[t]

    # Get mask
    mask = 255 * np.ones([self.tile_size, self.tile_size], np.uint8)

    is_before = self.data.mask_time <= tile_date
    if np.any(is_before):
      last_before = np.arange(len(is_before))[is_before][-1]
      later_mask = self.data.Mask[last_before, 0, y0:y1, x0:x1]
      mask = np.where(later_mask == 1, 1, mask)

    is_after  = self.data.mask_time >= tile_date
    if np.any(is_after):
      first_after = np.arange(len(is_after))[is_after][0]
      later_mask = self.data.Mask[first_after, 0, y0:y1, x0:x1]
      mask = np.where(later_mask == 0, 0, mask)

    tile = np.clip(np.nan_to_num(tile.values) / 10000, 0, 1)
    return torch.from_numpy(tile), torch.from_numpy(mask)

  def __len__(self):
    return self.T * self.H_tile * self.W_tile


def get_loader(scenes, batch_size, tile_size, sampling_mode, augment=False, augment_types=None, shuffle=False,
        num_workers=0, data_sources=None, data_root=None):
    scenes = [TimeseriesDataset(f'data/s2_timeseries/{scene}.nc', tile_size=tile_size, sampling_mode=sampling_mode) for scene in scenes]
    all_data = ConcatDataset(scenes)
    all_data = Augment(all_data, augment_types=augment_types)
    return DataLoader(all_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def get_vis_loader(vis_config, batch_size, tile_size=128, data_sources=None, data_root=None):
    vis_names = []
    vis_datasets = []
    for scene, indices in vis_config.items():
        dataset = TimeseriesDataset(f'data/s2_timeseries/{scene}.nc',
                                    tile_size=tile_size, sampling_mode='grid')
        vis_names += [f'{scene}-{i}' for i in indices]
        print('Subsetting:', len(dataset), '@', indices)
        filtered = Subset(dataset, indices)
        vis_datasets.append(filtered)
    vis_data = ConcatDataset(vis_datasets)

    loader = DataLoader(vis_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    return loader, vis_names


if __name__ == '__main__':
    from sys import argv
    from einops import rearrange
    from PIL import Image
    from lib.utils.plot_info import grid

    file = Path(argv[1])
    dataset = TimeseriesDataset(file, sampling_mode='grid')
    # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    for i in tqdm(range(len(dataset))):
        img, mask = dataset[i]
        if not (mask == 1).any():
            continue
        img  = img.numpy()
        mask = mask.numpy()

        mask = np.where(mask == 255, np.uint8(127),
               np.where(mask == 1,   np.uint8(255),
                                     np.uint8(  0)))

        print(img.min(), img.max(), end=' -> ')
        img = rearrange(img[[3,2,7]], 'C H W -> H W C')
        img  = np.clip(2 * 255 * img, 0, 255).astype(np.uint8)
        print(img.min(), img.max())
        combined = Image.fromarray(grid([[img, mask]]))
        combined.save(f'img/{file.stem}_{i}.jpg')
