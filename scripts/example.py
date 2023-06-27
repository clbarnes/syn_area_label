#!/usr/bin/env python3
from pathlib import Path
import json
import logging

import h5py
import numpy as np
import pandas as pd
import pymaid

from syn_area_label.bigcat_io import Annotations, LabelVolume
from syn_area_label.utils import (
    WorldCoord,
    project_presyn,
    roi_containing_rois,
    split_into_rois,
    setup_logging,
    write_table_homogeneous,
)
from syn_area_label.zarr_io import RoiExtractor, WrappedVolume, n5_to_dask

setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

# your data here
skids = [
    d["skeleton_id"] for d in json.loads(HERE.joinpath("kc_examples.json").read_text())
]
creds = json.loads(HERE.parent.joinpath("credentials/seymour.json").read_text())
cm = pymaid.CatmaidInstance(**creds)

neurons = pymaid.get_neurons(skids)

postsyn_nodes: pd.DataFrame
"""
columns "node_id", "z", "y", "x", like in navis
"""

connector_df: pd.DataFrame
"""
columns "connector_id", "node_id", "z", "y", "x", like in navis.
However, ensure that only postsynaptic nodes are included.
"""

postsyn_nodes = neurons.nodes
connector_df = neurons.postsynapses

# configuration

pre_site_creation_overshoot = 0.1
padding = 300  # nm
n5_url = "https://neurophyla.mrc-lmb.cam.ac.uk/tiles/0111-8/seymour.n5"
n5_ds = "volumes/raw/c0/s0"
translation = WorldCoord(6050, 0, 0)
resolution = WorldCoord(50, 3.8, 3.8)


# todo: write table of
# original connector id, new pre id, post tnid
# possibly pre skid, pre tnid, post skid? but these can all be determined post-hoc
all_locs, all_pre_post, all_pre_conn = project_presyn(postsyn_nodes, connector_df)

raw_arr = n5_to_dask(n5_url, n5_ds)
wrapped_vol = WrappedVolume(raw_arr, translation, resolution)

split_rois = split_into_rois(all_locs, all_pre_post, padding)
logger.info("Split ROIs into %s files", len(split_rois))

for idx, (locs, pre_post, rois) in enumerate(split_rois):
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

    conn_map_rows = []
    for pre_id in pre_post:
        conn_id = all_pre_conn[pre_id]
        conn_map_rows.append([pre_id, conn_id])
    conn_map_df = pd.DataFrame(
        conn_map_rows, columns=["presynaptic_site_id", "connector_id"], dtype=np.uint64
    )

    with h5py.File(HERE / f"data_{idx:02}.hdf5", "x") as f:
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
        clefts.to_hdf5(
            f, {"chunks": tuple(c[0] for c in ds.chunks), "compression": "lzf"}
        )
        write_table_homogeneous(f, conn_map_df, "connector_ids", np.uint64)
