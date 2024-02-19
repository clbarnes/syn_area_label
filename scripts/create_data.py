#!/usr/bin/env python3
from pathlib import Path
import logging

from syn_area_label import EdgeTables, cache_rois
import pymaid
from pymaid.stack import Stack

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Instantiate the CATMAID client.
# This example is public; see pymaid docs for authenticating with non-public instances.
cm = pymaid.CatmaidInstance("https://l1em.catmaid.virtualflybrain.org", None)

# Select a stack by name (visible in CATMAID UI).
stack = Stack.from_catmaid("L1 CNS 0-tilt")

# Select a stack mirror by name (visible in CATMAID UI).
# See `stack.set_mirror_auth` method for authenticating with non-public mirrors.
# A nearby N5 or neuroglancer stack mirror will probably be fastest.
stack.set_mirror("VFB")

# Use the scale 0 (maximum resolution) stack
volume = stack.get_scale(0)

# Find all of the edges whose postsynapses are on the given skeletons.
# See `EdgeTables.from_nodes` if you already know the connector and postsynaptic treenode IDs of interest.
etables = EdgeTables.from_skeletons(
    post_skeleton_ids=[
        16015738,
        16627773,
        16629931,
        14522124,
    ]
)

out_dir = Path(__file__).resolve().parent.parent.joinpath("data/output/large/examples")

# Generate the CREMI files
cache_rois(volume, etables, out_dir)
