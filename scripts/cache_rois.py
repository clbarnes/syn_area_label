#!/usr/bin/env python3
from pathlib import Path
import json
import logging

import pymaid

from syn_area_label.utils import (
    setup_logging,
)
from syn_area_label.roi_download import OffsetMultiscale, RoiCacher
from syn_area_label.synapses import EdgeTables

setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent


padding = 500  # nm
n5_url = "https://neurophyla.mrc-lmb.cam.ac.uk/tiles/0111-8/seymour.n5"
n5_ds = "volumes/raw/c0"

# your data here
skids = [
    16015738,
    16627773,
    16629931,
    14522124,
]
creds = json.loads(HERE.parent.joinpath("credentials/seymour.json").read_text())
cm = pymaid.CatmaidInstance(**creds)

xarr = OffsetMultiscale.from_paths(n5_url, n5_ds, offset={"z": 6050, "y": 0, "x": 0})[0]

etables = EdgeTables.from_skeletons(post_skeleton_ids=skids, remote_instance=cm)

cacher = RoiCacher(
    xarr,
    etables,
    HERE.parent / "data/output/large/kcs_cremi",
    padding,
    cremi=True,
)
cacher.write_all(False)
