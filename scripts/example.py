#!/usr/bin/env python3
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from syn_area_label.bigcat_io import Annotations, LabelVolume
from syn_area_label.utils import (
    WorldCoord,
    project_presyn,
    roi_containing_rois,
    split_into_rois,
)
from syn_area_label.zarr_io import RoiExtractor, WrappedVolume, n5_to_dask

# your data here

postsyn_nodes: pd.DataFrame
"""
columns "node_id", "z", "y", "x", like in navis
"""

connector_df: pd.DataFrame
"""
columns "connector_id", "node_id", "z", "y", "x", like in navis.
However, ensure that only postsynaptic nodes are included.
"""

# configuration

pre_site_creation_overshoot = 0.1
padding = 300  # nm
n5_url = "https://neurophyla.mrc-lmb.cam.ac.uk/tiles/0111-8/seymour.n5"
n5_ds = "volumes/raw/c0/s0"
translation = WorldCoord(6050, 0, 0)
resolution = WorldCoord(50, 3.8, 3.8)

here = Path(__file__).resolve().parent

all_locs, all_pre_post = project_presyn(postsyn_nodes, connector_df)

raw_arr = n5_to_dask(n5_url, n5_ds)
wrapped_vol = WrappedVolume(raw_arr, translation, resolution)

for idx, (locs, pre_post, rois) in enumerate(
    split_into_rois(all_locs, all_pre_post, padding)
):
    offset, shape = roi_containing_rois(rois)
    vox_offset = offset.to_voxel(resolution, translation, np.floor)
    actual_offset = vox_offset.to_world(resolution, translation)
    vox_shape = WorldCoord(*(offset.to_ndarray() + shape.to_ndarray())).to_voxel(
        resolution, translation, np.ceil
    )
    actual_offset_arr = actual_offset.to_ndarray()

    offset_locs = {
        k: WorldCoord(*(v.to_ndarray() - actual_offset_arr)) for k, v in locs.items()
    }

    with h5py.File(here / "data_{idx:02}.hdf5", "x") as f:
        extractor = RoiExtractor(wrapped_vol, vox_offset, vox_shape, f)
        ds = None
        for this_offset, this_shape in rois:
            ds = extractor.extract_roi(this_offset, this_shape, ds)

        annotations = Annotations.from_partners(offset_locs, pre_post)
        annotations.to_hdf5(f)

        if ds is None:
            continue

        clefts = LabelVolume(
            "clefts", np.zeros(ds.shape, dtype=np.uint64), resolution, actual_offset
        )
        clefts.to_hdf5(f, {"chunks": ds.chunks, "compression": "lzf"})
