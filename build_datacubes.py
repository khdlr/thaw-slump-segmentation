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

import pandas as pd
import geopandas as gpd
from joblib import Parallel, delayed

from lib import data
from lib.utils import init_logging, get_logger

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

def build_sentinel2_timeseries(scene_info, out_dir: Path):
    data.init_data_paths(out_dir.parent)
    start_date = pd.to_datetime('2010-01-01')
    end_date = pd.to_datetime('2030-01-01')
    # end_date = scene_info.image_date + pd.to_timedelta('5 days')
    scenes = data.Sentinel2.build_scenes(
        bounds=scene_info.geometry, crs=None, 
        start_date=start_date, end_date=end_date,
        prefix=scene_info.image_id
    )
    for scene in scenes:
        scene.save(out_dir / f'{scene.id}.id')


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
                build_sentinel2_timeseries(info, out_dir)
        else:
            Parallel(n_jobs=args.n_jobs)(
                delayed(build_sentinel2_timeseries)(info, out_dir) for _, info in scene_info.iterrows())
