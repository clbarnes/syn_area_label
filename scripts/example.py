#!/usr/bin/env python3
from pathlib import Path
import json
import logging

import h5py
import numpy as np
import pandas as pd
import pymaid
from tqdm import tqdm

from syn_area_label.bigcat_io import Annotations
from syn_area_label.utils import (
    WorldCoord,
    project_presyn,
    roi_containing_rois,
    split_into_rois,
    setup_logging,
    write_table_homogeneous,
    sanitise_chunks,
)
from syn_area_label.zarr_io import RoiExtractor, WrappedVolume, n5_to_dask

setup_logging(level=logging.INFO)
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

all_nodes = neurons.nodes
output_conn_ids = set(neurons.presynapses["connector_id"])
connector_df = neurons.postsynapses[
    neurons.postsynapses["connector_id"].isin(output_conn_ids)
]

# configuration

pre_site_creation_overshoot = 0.1
padding = 300  # nm
n5_url = "https://neurophyla.mrc-lmb.cam.ac.uk/tiles/0111-8/seymour.n5"
n5_ds = "volumes/raw/c0/s0"
translation = WorldCoord(6050, 0, 0)
resolution = WorldCoord(50, 3.8, 3.8)
out_dir = Path("/data/syn_area_volumes")


# todo: write table of
# original connector id, new pre id, post tnid
# possibly pre skid, pre tnid, post skid? but these can all be determined post-hoc
all_locs, all_pre_post, all_pre_conn = project_presyn(all_nodes, connector_df)

raw_arr = n5_to_dask(n5_url, n5_ds)
wrapped_vol = WrappedVolume(raw_arr, translation, resolution)

split_rois = split_into_rois(all_locs, all_pre_post, padding)
logger.info("Split ROIs into %s files", len(split_rois))

for idx, (locs, pre_post, rois) in enumerate(tqdm(split_rois, "output files")):
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

    with h5py.File(out_dir / f"data_{idx:02}.hdf5", "x") as f:
        annotations = Annotations.from_partners(offset_locs, pre_post)
        annotations.to_hdf5(f)

        extractor = RoiExtractor(wrapped_vol, vox_offset, vox_shape, f)
        ds = None
        counter = 0
        for this_offset, this_shape in tqdm(rois, "ROIs"):
            ds = extractor.extract_roi(this_offset, this_shape, ds)
            # todo: remove
            # counter += 1
            # if counter > 3:
            #     break

        if ds is None:
            continue

        labels = f.require_group("volumes").require_group("labels")
        chunks = sanitise_chunks(ds.chunks)
        label_ds = labels.create_dataset(
            "clefts", ds.shape, np.uint64, chunks=chunks, compression="lzf", fillvalue=0
        )
        label_ds.attrs["resolution"] = resolution.to_ndarray()
        label_ds.attrs["offset_from_project"] = actual_offset.to_ndarray()

        write_table_homogeneous(
            f, conn_map_df, "/annotations/presynaptic_site/pre_to_conn", np.uint64
        )
