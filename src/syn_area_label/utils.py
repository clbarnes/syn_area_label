from __future__ import annotations

import logging
from typing import Callable, Iterable, NamedTuple, Optional

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import DTypeLike

logger = logging.getLogger(__name__)


def setup_logging(level=logging.DEBUG):
    """Sane default logging setup.

    Should only be called once, and only by a script.
    """
    logging.basicConfig(level=level)

    # these are packages with particularly noisy logging
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


Rounder = Callable[[np.ndarray], np.ndarray]


class WorldCoord(NamedTuple):
    """ZYX coordinate of floats"""

    z: float
    y: float
    x: float

    def to_ndarray(self, dtype: DTypeLike = np.float64) -> np.ndarray:
        return np.array(self, dtype=dtype)

    def to_voxel(
        self, resolution: WorldCoord, offset: WorldCoord, round: Rounder = np.round
    ):
        frac = (self.to_ndarray() - offset.to_ndarray()) / resolution.to_ndarray()
        return VoxelCoord(*round(frac).astype(np.uint64))


class VoxelCoord(NamedTuple):
    """ZYX coordinate of integers"""

    z: int
    y: int
    x: int

    def to_ndarray(self, dtype: DTypeLike = np.uint64) -> np.ndarray:
        return np.array(self, dtype=dtype)

    def to_world(self, resolution: WorldCoord, offset: WorldCoord):
        return WorldCoord(
            *(
                self.to_ndarray(np.float64) * resolution.to_ndarray()
                + offset.to_ndarray()
            )
        )


def to_bbox(offset: WorldCoord, shape: WorldCoord):
    """Turn offset-shape pair into [[zmin ymin xmin], [zmax ymax xmax]"""
    start = offset.to_ndarray()
    end = start + shape.to_ndarray()
    return np.array([start, end])


def volume_of_intersection(
    os1: tuple[WorldCoord, WorldCoord], os2: tuple[WorldCoord, WorldCoord]
) -> int:
    """Find volume of intersection between two offset-shape pairs"""
    bboxes = np.array([to_bbox(*os1), to_bbox(*os2)])
    max_of_min = np.max(bboxes[:, 0], axis=0)
    min_of_max = np.min(bboxes[:, 1], axis=0)
    overlaps = min_of_max - max_of_min
    return np.prod(overlaps[overlaps >= 0])


def intersecting_volumes(
    offset_shapes: list[tuple[WorldCoord, WorldCoord]], threshold: float = 0
) -> Iterable[set[int]]:
    """Find which offset-shape pairs overlap, given a threshold volume."""
    g = nx.Graph()
    g.add_nodes_from(range(len(offset_shapes)))
    bboxes = [to_bbox(*os) for os in offset_shapes]

    pair = np.zeros((2, 2, 3), dtype=np.float64)
    for idx1 in range(len(offset_shapes) - 1):
        pair[0] = bboxes[idx1]
        for idx2 in range(idx1 + 1, len(offset_shapes)):
            pair[1] = bboxes[idx2]

            max_of_min = np.max(pair[:, 0], axis=0)
            min_of_max = np.min(pair[:, 1], axis=0)
            overlaps = min_of_max - max_of_min
            # todo: check this
            if np.all(overlaps > threshold):
                g.add_edge(idx1, idx2)

    for cc in nx.connected_components(g):
        yield set(cc)


def roi_containing_points(
    points: Iterable[WorldCoord], pad: Optional[WorldCoord] = None
) -> Optional[tuple[WorldCoord, WorldCoord]]:
    """Return a single offset-shape pair which contains all of the given points."""
    rows = []
    for p in points:
        rows.append(p.to_ndarray())

    if not rows:
        return None

    arr = np.array(rows, dtype=rows[-1].dtype)
    mins = np.min(arr, axis=0)
    maxes = np.max(arr, axis=0)
    if pad is not None:
        padding = pad.to_ndarray()
        mins -= padding
        maxes += padding
    shape = maxes - mins
    return WorldCoord(*mins), WorldCoord(*shape)


def roi_containing_rois(
    rois: Iterable[tuple[WorldCoord, WorldCoord]]
) -> tuple[WorldCoord, WorldCoord]:
    corners = []
    for offset, shape in rois:
        corners.append(offset.to_ndarray())
        corners.append(shape.to_ndarray + corners[-1])
    corn = np.array(corners, dtype=corners[-1].dtype)
    mi = corn.min(0)
    ma = corn.max(0)
    return WorldCoord(*mi), WorldCoord(*(ma - mi))


def project_presyn(
    postsyn_nodes: pd.DataFrame, connector_df: pd.DataFrame, xy_overshoot: float = 0.1
) -> tuple[dict[int, WorldCoord], dict[int, int]]:
    """Create presynaptic sites for the given postsynapses.

    Projects back from the postsynapse to the connector,
    then overshoots by a fraction of that distance in XY.
    This ensures that each postsynapse has a unique presynaptic site,
    which is likely to be inside the presynaptic cell,
    and which is is likely not to overlap with other presynaptic sites.

    Parameters
    ----------
    postsyn_nodes : pd.DataFrame
        columns "node_id", "z", "y", "x",
        a subset of the format navis uses.
    connector_df : pd.DataFrame
        columns "connector_id", "node_id", "z", "y", "x",
        a subset of the format navis uses.
        If you use a table from navis, you should first filter
        the dataframe so that only treenodes postsynaptic to the connector
        are included.
    xy_overshoot : float, optional
        The proportion of the postsyn-connector distance
        to project the new presynaptic site back (default 0.1).

    Returns
    -------
    locations : dict[int, WorldCoord]
        Mapp node IDs to locations
    pre_post : dict[int, int]
        Map presynaptic site IDs to postsynaptic side IDs
    """
    df = postsyn_nodes.merge(connector_df, on="node_id", suffixes=(None, "_c"))

    yx = ["y", "x"]
    yx_c = ["y_c", "x_c"]
    conn_yx = df[yx_c].to_numpy()
    yx_dists = conn_yx - df[yx].to_numpy()
    new_yx = conn_yx + yx_dists * xy_overshoot
    df[["y_n", "x_n"]] = new_yx

    locs = dict()
    pre_post = dict()

    pre_id = 0
    used_ids = set(postsyn_nodes["node_id"])

    for row in df.itertuples(index=False):
        while pre_id in used_ids:
            pre_id += 1
        if pre_id not in locs:
            locs[pre_id] = WorldCoord(row.z_c, row.y_n, row.x_n)
            pre_id += 1

        if row.node_id not in locs:
            locs[row.node_id] = WorldCoord(row.z, row.y, row.x)
        pre_post[pre_id] = row.node_id

    return locs, pre_post


def split_into_rois(
    locs: dict[int, WorldCoord],
    pre_post: dict[int, int],
    pad: float,
) -> list[
    tuple[
        dict[int, WorldCoord],
        dict[int, int],
        list[tuple[WorldCoord, WorldCoord]],
    ]
]:
    """Split connectors into groups of those with overlapping ROIs.

    Returns a list of tuples where the first
    """
    pre_post_list = list(pre_post.items())
    offset_shapes = []
    padding = WorldCoord(pad, pad, pad)
    for pre_id, post_id in pre_post_list:
        offset_shape = roi_containing_points([locs[pre_id], locs[post_id]], padding)
        if offset_shape is None:
            raise RuntimeError("impossible")
        offset_shapes.append(offset_shape)

    out = []

    for cc in intersecting_volumes(offset_shapes):
        loc = dict()
        partners = dict()
        rois = []
        for idx in sorted(cc):
            rois.append(offset_shapes[idx])
            pre_id, post_id = pre_post_list[idx]
            partners[pre_id] = post_id
            loc[pre_id] = locs[pre_id]
            loc[post_id] = locs[post_id]
        out.append((loc, partners, rois))

    return out
