from __future__ import annotations
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import math
from os import cpu_count
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple, Optional, Sequence
from multiscale_read import NglN5Multiscale
import zarr
import xarray as xr
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from functools import cache
import networkx as nx
import itertools
from .utils import CompletionChecker, DfBuilder, Dim, project_presyn

from syn_area_label.synapses import EdgeTables


class OffsetMultiscale(NglN5Multiscale):
    def __init__(
        self, group: zarr.Group, offset: Optional[dict[str, float]] = None
    ) -> None:
        super().__init__(group)
        self.offset = offset

    @classmethod
    def from_paths(
        cls, container, group: str, store_kwargs=None, group_kwargs=None, offset=None
    ):
        if store_kwargs is None:
            store_kwargs = dict()
        if group_kwargs is None:
            group_kwargs = dict()

        store = zarr.N5FSStore(container, **store_kwargs)
        group_kwargs.setdefault("mode", "r")
        root = zarr.open_group(store, **group_kwargs)
        return cls(root[group], offset)

    def _get_item(self, idx: int) -> xr.DataArray:
        xarr = super()._get_item(idx)
        if self.offset is not None:
            new_coords = {k: v + self.offset.get(k, 0) for k, v in xarr.coords.items()}
            xarr = xarr.assign_coords(new_coords)
        return xarr


@dataclass
class RoiMetadata:
    dimensions: list[Dim]
    translation: dict[Dim, float]
    resolution: dict[Dim, float]

    def translation_ordered(self, order=None) -> list[float]:
        if order is None:
            order = self.dimensions
        return [self.translation[k] for k in order]

    def resolution_ordered(self, order=None) -> list[float]:
        if order is None:
            order = self.dimensions
        return [self.resolution[k] for k in order]


class RoiCacher:
    """Class for writing ROIs around connectors into CREMI files."""

    def __init__(
        self,
        volume: xr.DataArray,
        edge_tables: EdgeTables,
        outdir: Path,
        padding: float = 300,
        cremi=True,
    ) -> None:
        """Instantiate the RoiCacher.

        Parameters
        ----------
        volume : xr.DataArray
            uint8 volume with coordinate arrays in world units,
            and the dimensions "x", "y", and "z".
            See the multiscale_read package and pymaid.Stack to create these easily.
        edge_tables : EdgeTables
            Tables of synaptic edges for inclusion (see `EdgeTables` class).
        outdir : Path
            Path to output directory.
        padding : float, optional
            How much to pad the ROI around the axis-aligned bounding box
            of the presynapse-connector-postsynapse points,
            in world units.
            By default 300
        cremi : bool, optional
            Whether to write a CREMI-format HDF5 file with some extra tables.
            Otherwise writes a directory of CSVs and the image volume in a numpy array.
            By default True.
        """
        self.volume = volume
        self.edge_tables = edge_tables
        self.outdir = Path(outdir)
        self.padding = padding
        self.cremi = cremi

        self.padding_dict = {k: padding for k in "zyx"}
        self.roi_table = self.edge_tables.rois(self.padding_dict)
        self.roi_dir = self.outdir / "rois"
        self.id_pad = len(str(len(self.edge_tables.edges)))

        self.checker = CompletionChecker(self.roi_dir)

    def _write_connector(self, conn_id: int, overwrite=True) -> int:
        ets = self.edge_tables.single_connector(conn_id)

        dpath = self.roi_dir / f"conn_{conn_id}"
        if not overwrite and self.checker.is_complete(dpath):
            return -1

        out = len(ets.edges)
        if out == 0:
            return out

        self.checker.mark_in_progress(dpath)

        slicing = dict()
        extents = ets.total_roi(self.padding_dict)
        for d, (mn, mx) in extents.items():
            slicing[d] = slice(mn, mx)

        subvol = self.volume.sel(slicing)

        meta = RoiMetadata(
            list(subvol.dims),
            {k: v[0] for k, v in subvol.coords.items()},
            {k: v[1] - v[0] for k, v in subvol.coords.items()},
        )
        roi = Roi(meta, ets, subvol.to_numpy())
        if self.cremi:
            roi.to_cremi(dpath.with_suffix(".hdf5"))
        else:
            roi.to_directory(dpath)

        self.checker.mark_complete(dpath)

        return out

    # def write_edge(self, edge_id: int, overwrite=True):
    #     dpath = self.roi_dir / f"edge_{edge_id:0{self.id_pad}}"
    #     if not overwrite and self.checker.is_complete(dpath):
    #         return False

    #     self.checker.mark_in_progress(dpath)
    #     # dpath.mkdir(exist_ok=True)
    #     # meta_path = dpath / "metadata.json"
    #     # if meta_path.is_file():
    #     #     return False

    #     df = self.roi_table

    #     rows = df[df["edge_id"] == edge_id]
    #     if len(rows) == 0:
    #         raise ValueError(f"No edge with ID {edge_id}")
    #     row = rows.iloc[0]

    #     slicing = dict()
    #     extents = dict()
    #     for d in "zyx":
    #         mn = row[f"{d}_min"]
    #         mx = row[f"{d}_max"]
    #         slicing[d] = slice(mn, mx)
    #         extents[d] = (mn, mx)

    #     subvol = self.volume.sel(slicing)

    #     meta = RoiMetadata(
    #         list(subvol.dims),
    #         {k: v[0] for k, v in subvol.coords.items()},
    #         {k: v[1] - v[0] for k, v in subvol.coords.items()},
    #     )
    #     out = Roi(meta, self.edge_tables.in_roi(extents), subvol.to_numpy())
    #     if self.cremi:
    #         out.to_cremi(dpath.with_suffix(".hdf5"))
    #     else:
    #         out.to_directory(dpath)
    #     self.checker.mark_complete(dpath)

    #     return True

    def _write_connectors_serial(self, conns: Sequence[int], overwrite: bool) -> int:
        count = 0
        for conn in tqdm(conns, "writing connector ROIs"):
            res = self._write_connector(conn, overwrite)
            if res > 0:
                count += 1
        return count

    def _write_connectors_threaded(
        self, conns: Sequence[int], overwrite: bool, threads: int
    ) -> int:
        threads = int(threads)
        if threads <= 1:
            threads = cpu_count() or 1

        count = 0
        with ThreadPoolExecutor(threads) as pool:
            for res in tqdm(
                pool.map(lambda c: self._write_connector(c, overwrite), conns),
                "writing connector ROIs",
                total=len(conns),
            ):
                if res > 0:
                    count += 1

        return count

    def write_connectors(
        self,
        overwrite: bool = True,  # threads: Optional[int] = None
    ) -> int:
        """Write one ROI per connector.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite existing ROIs at the output path, by default True

        Returns
        -------
        int
            How many ROIs were written.
        """
        self._write_shared()

        conns = self.edge_tables.connectors["connector_id"].unique()
        return self._write_connectors_serial(conns, overwrite)
        # return self._write_connectors_threaded(conns, overwrite, threads)

    # def write_all(self, overwrite=True):
    #     self._write_shared()

    #     count = 0

    #     for edge_id in tqdm(self.edge_tables.edges["edge_id"], "writing ROIs"):
    #         count += self.write_edge(edge_id, overwrite)
    #     return count

    def _write_shared(self):
        self.roi_dir.mkdir(parents=True, exist_ok=True)
        self.edge_tables.to_directory(self.outdir / "tables", self.padding_dict)


def cache_rois(
    volume: xr.DataArray,
    edge_tables: EdgeTables,
    outdir: Path,
    padding: float = 300,
    # threads=None,
):  # noqa: F821
    """Write CREMI files for the given edge tables.

    Produces one CREMI file per connector node,
    which may contain several edges.

    Parameters
    ----------
    volume : xr.DataArray
        Data array with dimensions "z", "y", and "x",
        and world-space coordinate arrays.
    edge_tables : EdgeTables
        Edges of interest, can be created from skeleton IDs or edges using pymaid.
    outdir : Path
        Directory under which to store output (should not exist).
    padding : float, optional
        In world units, how far outside the bounding box of the nodes involved in an edge should the ROI cover,
        by default 300
    """
    cacher = RoiCacher(volume, edge_tables, outdir, padding, True)
    cacher.write_connectors()


class Bbox(NamedTuple):
    z_min: float
    z_max: float
    y_min: float
    y_max: float
    x_min: float
    x_max: float

    @classmethod
    def from_mapping(cls, d: Mapping):
        args = []
        for dim in "zyx":
            for extent in ["_min", "_max"]:
                args.append(d[dim + extent])
        return cls(*args)

    def validate(self):
        for mn, mx in self.extents():
            assert mn <= mx

    def extents(self) -> Iterable[tuple[float, float]]:
        for d in "zyx":
            yield (
                getattr(self, f"{d}_min"),
                getattr(self, f"{d}_max"),
            )

    @cache
    def volume(self):
        prod = 1
        for mn, mx in self.extents():
            prod *= mx - mn
            if prod <= 0:
                return 0
        return prod

    @cache
    def union(self, other: Bbox) -> Bbox:
        args = []
        for (mn1, mx1), (mn2, mx2) in zip(self.extents(), other.extents()):
            args.append(min(mn1, mn2))
            args.append(max(mx1, mx2))
        return Bbox(*args)

    @cache
    def intersection(self, other: Bbox) -> Optional[Bbox]:
        args = []
        for (mn1, mx1), (mn2, mx2) in zip(self.extents(), other.extents()):
            if mx1 < mn2 or mx2 < mn1:
                return None

            args.append(max(mn1, mn2))
            args.append(min(mx1, mx2))
        return Bbox(*args)


class BboxNode(ABC):
    bbox: Bbox

    def __hash__(self) -> int:
        return hash(self.bbox)

    def union(self, other: BboxNode, union_bbox=None) -> BboxBranch:
        return BboxBranch(self, other, union_bbox)


class BboxLeaf(BboxNode):
    def __init__(self, bbox: Bbox) -> None:
        self.bbox = bbox

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, BboxLeaf):
            bb = __o.bbox
        else:
            bb = __o
        return self.bbox == bb


def node_tuple(bb1: BboxNode, bb2: BboxNode):
    if bb1.bbox > bb2.bbox:
        return (bb2, bb1)
    return (bb1, bb2)


class BboxBranch(BboxNode):
    def __init__(
        self,
        bb1: BboxNode,
        bb2: BboxNode,
        union_bbox: Optional[Bbox] = None,
    ):
        self.children = node_tuple(bb1, bb2)

        if union_bbox is None:
            union_bbox = bb1.bbox.union(bb2.bbox)
        self.bbox: Bbox = union_bbox

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, BboxBranch):
            return False
        return self.children == __o.children

    def filler_volume(self):
        v = self.bbox.volume()
        for b in self.children:
            v -= b.bbox.volume()
        return v


class RoiClusters:
    def __init__(self, rois: list[Bbox], resolution=None) -> None:
        # if isinstance(rois, EdgeTables):
        #     rois = rois.rois({"x": 0, "y": 0, "z": 0})

        # if resolution is not None:
        #     rois = rois.copy(True)
        #     for d, r in resolution.items():
        #         rois[f"{d}_min"] /= r
        #         rois[f"{d}_max"] /= r

        # self.bbox_to_id = {
        #     Bbox.from_mapping(row._asdict()): row.edge_id
        #     for row in rois.itertuples(index=False)
        # }
        self.dists = nx.Graph()
        nodes = [BboxLeaf(b) for b in rois]
        self.bbox_to_id = {roi: idx for idx, roi in enumerate(nodes)}
        self.dists.add_nodes_from(nodes, filler=0)
        for b1, b2 in itertools.combinations(nodes, 2):
            branch = BboxBranch(b1, b2)
            self.dists.add_edge(b1, b2, branch=branch, filler=branch.filler_volume())

    def _lowest_cluster(self) -> bool:
        min_tup = (None, None, None, math.inf)
        for b1, b2, data in self.dists.edges(data=True):
            branch = data["branch"]
            fvol = branch.filler_volume()
            if fvol < min_tup[-1]:
                min_tup = (b1, b2, branch, fvol)

        if min_tup[0] is None:
            return False

        b1, b2, branch, _ = min_tup

        self.dists.remove_nodes_from((b1, b2))
        for node in list(self.dists.nodes()):
            branch2 = BboxBranch(node, branch)
            self.dists.add_edge(
                node, branch, branch=branch2, filler=branch2.filler_volume()
            )

        return True

    def _linkage(self, node: BboxNode, rows=None):
        if rows is None:
            rows = []
        if isinstance(node, BboxLeaf):
            pass
        elif isinstance(node, BboxBranch):
            b1, b2 = node.children
            self._linkage(b1, rows)
            self._linkage(b2, rows)

            next_id = len(self.bbox_to_id)
            rows.append(
                [
                    self.bbox_to_id[b1],
                    self.bbox_to_id[b2],
                    node.filler_volume(),
                    next_id,
                ]
            )
            self.bbox_to_id[node] = next_id

    def cluster(self) -> pd.DataFrame:
        while self._lowest_cluster():
            pass

        if len(self.dists) != 1:
            raise RuntimeError(f"expected 1 node, got {len(self.dists)}")
        n = next(self.dists.nodes())

        linkage = self._linkage(n)
        return pd.DataFrame(
            linkage, columns=["id1", "id2", "filler_volume", "new_id"]
        ).astype(
            {
                "id1": np.uint64,
                "id2": np.uint64,
                "filler_volume": np.float64,
                "new_id": np.uint64,
            }
        )


class Roi:
    def __init__(
        self,
        metadata: RoiMetadata,
        edge_tables: EdgeTables,
        volume: np.ndarray,
    ) -> None:
        self.metadata = metadata
        self.edge_tables = edge_tables
        self.volume = volume

    def to_directory(self, dpath: Path):
        dpath.mkdir(exist_ok=False, parents=True)
        self.edge_tables.to_directory(dpath / "tables")
        eids = list(self.edge_tables.edges["edge_id"])
        np.save(dpath / "volume.npy", self.volume)

        meta = asdict(self.metadata)
        meta["edge_ids"] = eids
        with dpath.joinpath("metadata.json").open("w") as f:
            json.dump(meta, f, indent=True, sort_keys=True)

    @classmethod
    def from_directory(cls, dpath: Path):
        etables = EdgeTables.from_directory(dpath / "tables")
        with dpath.joinpath("metadata.json").open() as f:
            all_meta = json.load(f)
        meta = {k: all_meta[v] for k, v in RoiMetadata.__dataclass_fields__}
        metadata = RoiMetadata(**meta)
        vol = np.load(dpath / "volume.npy")
        return cls(metadata, etables, vol)

    def to_cremi(self, fpath: Path):
        from cremi import Annotations, Volume, CremiFile

        site_ids = None

        with CremiFile(fpath, "w") as f:
            raw = Volume(self.volume, resolution=self.metadata.resolution_ordered())
            f.write_raw(raw)

            clefts = Volume(
                np.zeros(self.volume.shape, np.uint64),
                resolution=self.metadata.resolution_ordered(),
            )
            f.write_clefts(clefts)

            et = self.edge_tables.translate(
                {k: -v for k, v in self.metadata.translation.items()}, True
            )
            tns = et.treenodes.rename(columns={"treenode_id": "node_id"}, inplace=False)
            post_tns = tns[tns["node_id"].isin(et.edges["treenode_id_post"])]
            edges = et.edges.rename(
                columns={"treenode_id_post": "node_id"}, inplace=False
            )
            conns = edges.merge(et.connectors, on="connector_id")
            locs, pre_post, pre_conn = project_presyn(post_tns, conns)
            builder = DfBuilder(
                {
                    "presynaptic_site_id": "uint64",
                    "connector_id": "uint64",
                    "treenode_id_post": "uint64",
                }
            )
            for pre_id in sorted(set(pre_post).intersection(pre_conn)):
                builder.append([pre_id, pre_conn[pre_id], pre_post[pre_id]])
            site_ids = builder.build()

            edge_ids = list(edges["edge_id"])
            annotations = Annotations()
            for pre_id, post_id in pre_post.items():
                e_pre_id = edge_ids[pre_id]
                # todo: check order
                annotations.add_annotation(
                    e_pre_id,
                    "presynaptic_site",
                    locs[pre_id].ordered(self.metadata.dimensions),
                )
                annotations.add_comment(e_pre_id, "")
                annotations.add_annotation(
                    post_id,
                    "postsynaptic_site",
                    locs[post_id].ordered(self.metadata.dimensions),
                )
                annotations.set_pre_post_partners(e_pre_id, post_id)

            f.write_annotations(annotations)
            f.h5file.attrs["offset_from_project"] = self.metadata.translation_ordered()
            f.h5file.attrs["dimensions"] = self.metadata.dimensions

        if site_ids is None:
            extra = dict()
        else:
            extra = {"site_ids": site_ids}
        self.edge_tables.to_hdf5(fpath, "tables", extra)
