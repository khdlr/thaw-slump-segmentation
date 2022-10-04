#!/usr/bin/env python
# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Usecase 2 Data Preprocessing Script
"""
import argparse
from datetime import datetime
from pathlib import Path
import os
from collections import defaultdict
import xarray
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from joblib import Parallel, delayed

from lib import data
from lib.utils import init_logging, get_logger
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n_jobs", default=-1, type=int, help="number of parallel joblib jobs")
parser.add_argument("--data_dir", default='data', type=Path, help="Path to data processing dir")
parser.add_argument("--log_dir", default='logs', type=Path, help="Path to log dir")
parser.add_argument("--mode", default='planet', choices=['planet', 'sentinel2', 's2_timeseries'],
        help="The type of data cubes to build.")


def complete_scene(scene):
    # scene.add_layer(data.RelativeElevation())
    scene.add_layer(data.AbsoluteElevation())
    scene.add_layer(data.Slope())
    scene.add_layer(data.TCVIS())


def build_planet_cube(planet_file: Path, out_dir: Path):
    # We need to re-initialize the data paths every time
    # because of the multi-processing
    data.init_data_paths(out_dir.parent)

    scene = data.PlanetScope.build_scene(planet_file)
    labels = gpd.read_file(next(planet_file.parent.glob('*.shp')))
    scene.add_layer(data.Mask(labels.geometry))
    complete_scene(scene)
    out_dir.mkdir(exist_ok=True, parents=True)
    scene.save(out_dir / f'{scene.id}.nc')


def build_sentinel2_cubes(scene_info, out_dir: Path):
    data.init_data_paths(out_dir.parent)
    start_date = scene_info.image_date - pd.to_timedelta('5 days')
    end_date = scene_info.image_date + pd.to_timedelta('5 days')
    scenes = data.Sentinel2.build_scenes(
        bounds=scene_info.geometry, crs=None, 
        start_date=start_date, end_date=end_date,
        prefix=scene_info.image_id
    )
    for scene in scenes:
        complete_scene(scene)
        scene.save(out_dir / f'{scene.id}.id')


def build_sentinel2_timeseries(site_poly, image_id, tiles, targets, out_dir: Path):
    data.init_data_paths(out_dir.parent)
    print(f'Starting build for scene {image_id}')
    print(f'Tiles: {len(tiles)}')
    print(f'Targets: {len(targets)}')
    for _, tile in tiles.iterrows():
      print(f'Tile {tile.image_id}: {(targets.image_id == tile.image_id).sum()}')

    # Build S2 Scenes
    # start_date = pd.to_datetime('2015-01-01')
    start_date = pd.to_datetime(targets.image_date).min() - pd.to_timedelta('14 days')
    end_date = pd.to_datetime('2022-09-20')
    scenes = data.Sentinel2.build_scenes(
        bounds=site_poly, crs='EPSG:3995', 
        start_date=start_date, end_date=end_date,
        prefix=image_id
    )

    x_scenes = defaultdict(list)
    for scene in tqdm(scenes):
        xarr = scene.to_xarray()
        date = xarr.Sentinel2.date
        xarr = xarr.expand_dims({'time': [date]}, axis=0)
        xarr['Sentinel2'] = xarr['Sentinel2'] / 10000
        x_scenes[scene.crs].append((scene, xarr))

    encoding = dict(Sentinel2={
      'scale_factor': 1/255, 'add_offset': 0, 'dtype': 'uint8','_FillValue': 0})
    for crs in x_scenes:
      # Build Masks
      sample_scene = x_scenes[crs][0][0]
      xmasks = []
      for i in tiles.index:
        tile_targets = targets[targets.image_id == tiles.image_id[i]]
        mask = data.Mask(tile_targets.geometry, tiles.loc[[i]].geometry)
        mask_scene = data.Scene(f'{tiles.image_id.loc[i]}_{crs}', crs,
            sample_scene.transform, sample_scene.size, layers=[mask])
        date = pd.to_datetime(tile_targets.image_date.iloc[0])
        xmask = mask_scene.to_xarray()
        xmask = xmask.expand_dims({'mask_time': [date]}, axis=0)
        xmasks.append(xmask)

      xarrs = [xarr for scene, xarr in x_scenes[crs]]
      print('xarrs')
      for x in xarrs:
        print(x.Sentinel2.shape, x.dims)
      print('xmasks')
      for x in xmasks:
        print(x.Mask.shape, x.dims)
      combined = xarray.combine_by_coords(xarrs + xmasks, combine_attrs='drop_conflicts')
      combined.to_netcdf(out_dir / f'{image_id}_{crs}.nc', engine='h5netcdf', encoding=encoding)

    print(f'Done with build for scene {image_id}')


if __name__ == "__main__":
    args = parser.parse_args()

    global DATA_ROOT, INPUT_DATA_DIR, BACKUP_DIR, DATA_DIR, AUX_DIR

    DATA_ROOT = Path(args.data_dir)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = Path(args.log_dir) / f'build_data_cubes-{timestamp}.log'
    if not Path(args.log_dir).exists():
        os.mkdir(Path(args.log_dir))
    init_logging(log_path)
    logger = get_logger('setup_raw_data')
    logger.info('################################')
    logger.info('# Starting Building Data Cubes #')
    logger.info('################################')

    if args.mode == 'planet':
        planet_files = list((DATA_ROOT / 'input').glob('*/*_SR*.tif'))
        out_dir = DATA_ROOT / 'planet_cubes'

        if args.n_jobs == 0:
            for planet_file in planet_files:
                build_planet_cube(planet_file, out_dir)
        else:
            Parallel(n_jobs=args.n_jobs)(
                delayed(build_planet_cube)(planet_file, out_dir) for planet_file in planet_files)
    elif args.mode == 'sentinel2':
        shapefile_root = DATA_ROOT / 'ML_training_labels' / 'retrogressive_thaw_slumps'
        out_dir = DATA_ROOT / 's2_cubes'

        labels = map(gpd.read_file, shapefile_root.glob('*/TrainingLabel*.shp'))
        labels = pd.concat(labels).reset_index(drop=True)
        labels['image_date'] = pd.to_datetime(labels['image_date'])
        id2date = dict(labels[['image_id', 'image_date']].values)

        scenes = map(gpd.read_file, shapefile_root.glob('*/ImageFootprints*.shp'))
        scenes = pd.concat(scenes).reset_index(drop=True)
        scenes = scenes[scenes.image_id.isin(labels.image_id)]
        scenes['image_date'] = scenes.image_id.apply(id2date.get)

        scene_info = scenes.loc[:, ['image_id', 'image_date', 'geometry']]
        if args.n_jobs == 0:
            for _, info in scene_info.iterrows():
                build_sentinel2_cubes(info, out_dir)
        else:
            Parallel(n_jobs=args.n_jobs)(
                delayed(build_sentinel2_cubes)(info, out_dir) for _, info in scene_info.iterrows())
    elif args.mode == 's2_timeseries':
        shapefile_root = DATA_ROOT / 'ML_training_labels' / 'retrogressive_thaw_slumps'
        out_dir = DATA_ROOT / 's2_timeseries'

        targets = []
        for p in shapefile_root.glob('*/TrainingLabel*.shp'):
            targets.append(gpd.read_file(p).to_crs('EPSG:3995'))
        targets = pd.concat(targets)
        targets['geometry'] = targets['geometry'].buffer(0)

        scenes = []
        for p in shapefile_root.glob('*/*Footprints*.shp'):
            scenes.append(gpd.read_file(p).to_crs('EPSG:3995'))
        scenes = pd.concat(scenes)
        scenes['geometry'] = scenes['geometry'].buffer(0)

        # Semijoin  targets and sites:
        targets = targets[targets.image_id.isin(scenes.image_id)]
        scenes = scenes[scenes.image_id.isin(targets.image_id)]

        sites = unary_union(scenes.geometry)
        sites = list(sites.geoms)

        site_info = []
        for site_poly in sites:
            tiles = scenes[scenes.intersects(site_poly)]
            site_id = tiles.iloc[tiles.area.argmax()].image_id
            site_info.append((site_poly, site_id, tiles, targets[targets.image_id.isin(tiles.image_id)]))

        if args.n_jobs == 0:
          for info in site_info[7:]:
            build_sentinel2_timeseries(*info, out_dir)
        else:
            Parallel(n_jobs=args.n_jobs)(
                delayed(build_sentinel2_timeseries)(*info, out_dir) for info in site_info)
