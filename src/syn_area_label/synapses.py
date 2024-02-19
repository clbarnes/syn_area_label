import logging
from pathlib import Path
from typing import NamedTuple, NotRequired, Optional, Self, TypedDict
from collections.abc import Iterator, Sequence
from copy import copy, deepcopy

import pymaid
import numpy as np
import pandas as pd

from .constants import DIMS
from .utils import DfBuilder, JsoPrimitive, Dim

logger = logging.getLogger(__name__)

edge_columns = {
    "edge_id": np.uint64,
    "treenode_id_pre": np.uint64,
    "connector_id": np.uint64,
    "treenode_id_post": np.uint64,
}
treenode_columns = {
    "treenode_id": np.uint64,
    "skeleton_id": np.uint64,
    "x": np.float64,
    "y": np.float64,
    "z": np.float64,
}
connector_columns = {
    "connector_id": np.uint64,
    "x": np.float64,
    "y": np.float64,
    "z": np.float64,
}
skeleton_columns = {"skeleton_id": np.uint64, "neuron_name": str}


def write_tsv(path: Path, df: pd.DataFrame):
    return df.to_csv(path, sep="\t", index=False)


def read_tsv(path: Path, dtype=None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=dtype)


def subset_dict(d, keys) -> dict:
    return {k: d[k] for k in keys}


class DfDict(TypedDict):
    data: dict[str, list[JsoPrimitive]]
    dtypes: dict[str, str]
    columns: NotRequired[list[str]]


def df_to_dict(df: pd.DataFrame) -> DfDict:
    return DfDict(
        data=df.to_dict("list"),
        dtypes={k: v.name for k, v in df.dtypes.to_dict().items()},
        columns=df.columns.tolist(),
    )


def dict_to_df(d: DfDict) -> pd.DataFrame:
    columns = d.get("columns", list(d["dtypes"]))
    df = pd.DataFrame.from_dict(d["data"], columns=columns, dtype=object)
    return df.astype(d["dtypes"])


class EdgeTables(NamedTuple):
    edges: pd.DataFrame
    """edge_id, treenode_id_pre, connector_id, treenode_id_post"""

    treenodes: pd.DataFrame
    """treenode_id, skeleton_id, x, y, z"""

    connectors: pd.DataFrame
    """connector_id, x, y, z"""

    skeletons: pd.DataFrame
    """skeleton_id, neuron_name"""

    def to_directory(self, path: Path, padding: Optional[dict[Dim, float]] = None):
        """Store tables as a directory of TSVs.

        Optionally include ROIs with the given amount of padding in each dimension.
        """
        path.mkdir(exist_ok=True, parents=True)
        for name, df in self._asdict().items():
            p = path / f"{name}.tsv"
            write_tsv(p, df)

        if padding is None:
            return

        df = self.rois(padding)
        write_tsv(path / "rois.tsv", df)

    @classmethod
    def from_directory(cls, path: Path) -> Self:
        """Read from a directory of TSVs"""
        return cls(
            read_tsv(path / "edges.tsv", edge_columns),
            read_tsv(path / "treenodes.tsv", treenode_columns),
            read_tsv(path / "connectors.tsv", connector_columns),
            read_tsv(path / "skeletons.tsv", skeleton_columns),
        )

    def to_dict(self, padding: Optional[dict[Dim, float]] = None) -> dict[str, DfDict]:
        """Convert to a dict of dicts (for e.g. JSON serialisation)."""
        d = {k: df_to_dict(v) for k, v in self._asdict().items()}
        if padding is not None:
            df = self.rois(padding)
            d["rois"] = df_to_dict(df)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, DfDict]) -> Self:
        """Read from a dict of dataframe-like dicts (e.g. from JSON serialisation)"""
        return cls(**{k: dict_to_df(d[k]) for k in cls._fields})

    def to_hdf5(
        self,
        hdf5_path,
        group: str = "",
        extra_tables: Optional[dict[str, pd.DataFrame]] = None,
    ):
        """Store in pytables format in a given HDF5 container"""
        if extra_tables is None:
            extra_tables = dict()
        with pd.HDFStore(hdf5_path, "a") as f:
            f[group + "/edges"] = self.edges
            f[group + "/treenodes"] = self.treenodes
            f[group + "/connectors"] = self.connectors
            f[group + "/skeletons"] = self.skeletons
            for k, df in extra_tables.items():
                f[f"{group}/{k}"] = df

    @classmethod
    def from_hdf5(cls, hdf5_path, group: str = ""):
        """Read from an HDF5 container."""
        with pd.HDFStore(hdf5_path, "r") as f:
            return cls(
                f[group + "/edges"],
                f[group + "/treenodes"],
                f[group + "/connectors"],
                f[group + "/skeletons"],
            )

    def merge_locations(self, neuron_names=False) -> pd.DataFrame:
        """Get a single table containing all locations referenced."""
        df = self.edges
        df = pd.merge(df, self.connectors, on="connector_id", suffixes=(None, "_conn"))
        df.rename(columns={d: d + "_conn" for d in "xyz"}, inplace=True)
        suffixes = ["_pre", "_post"]
        for suff in suffixes:
            df = pd.merge(
                df,
                self.treenodes,
                left_on="treenode_id" + suff,
                right_on="treenode_id",
                suffixes=(None, suff),
            )
            df.drop(columns="treenode_id", inplace=True)
            df.rename(
                columns={pre: pre + suff for pre in ["skeleton_id", "x", "y", "z"]},
                inplace=True,
            )
            if neuron_names:
                df = pd.merge(
                    df,
                    self.skeletons,
                    left_on="skeleton_id" + suff,
                    right_on="skeleton_id",
                    suffixes=(None, suff),
                )
                df.drop(columns="skeleton_id", inplace=True)
                df.rename(
                    columns={"neuron_name": "neuron_name" + suff},
                    inplace=True,
                )
        return df

    def rois(self, padding: dict[Dim, float]) -> pd.DataFrame:
        """Get the ROIs of each edge."""
        merged = self.merge_locations()
        suffixes = ["_conn", "_pre", "_post"]
        new_cols = dict()
        axis = 1  # todo: check axis
        for d in ["x", "y", "z"]:
            xyz = merged[[f"{d}{suff}" for suff in suffixes]]
            new_cols[f"{d}_min"] = xyz.min(axis=axis) - padding[d]
            new_cols[f"{d}_max"] = xyz.max(axis=axis) + padding[d]

        return pd.DataFrame.from_dict({"edge_id": merged["edge_id"], **new_cols})

    def total_roi(self, padding: dict[Dim, float]) -> dict[Dim, tuple[float, float]]:
        """Get a single ROI covering all nodes"""
        out = dict()
        for d in DIMS:
            mn = min(self.connectors[d].min(), self.treenodes[d].min())
            mx = max(self.connectors[d].max(), self.treenodes[d].max())
            out[d] = (mn - padding[d], mx + padding[d])
        return out

    def single_connector(self, connector_id: int) -> Self:
        """Cut tables down to only those involving the given connector"""
        conns = self.connectors[self.connectors["connector_id"] == connector_id]
        edges = self.edges[self.edges["connector_id"] == connector_id]
        tn_set = set(edges["treenode_id_pre"]).union(edges["treenode_id_post"])
        treenodes = self.treenodes[self.treenodes["treenode_id"].isin(tn_set)]
        skids = set(treenodes["skeleton_id"])
        skeletons = self.skeletons[self.skeletons["skeleton_id"].isin(skids)]
        return type(self)(
            edges.copy(), treenodes.copy(), conns.copy(), skeletons.copy()
        )

    def by_connector(self) -> Iterator[tuple[int, Self]]:
        """Yield an EdgeTables instance for each connector in this instance"""
        conns = self.connectors["connector_id"].unique()
        for c in conns:
            yield c, self.single_connector(c)

    def in_roi(self, extents: dict[Dim, tuple[float, float]], copy=True) -> Self:
        """Cut tables down to only those where all nodes are in the ROI"""
        treenodes = self.treenodes
        connectors = self.connectors

        # eliminate tree&connector nodes outside of ROI
        for d, (mi, ma) in extents.items():
            treenodes = treenodes.loc[
                np.logical_and(treenodes[d] > mi, treenodes[d] < ma)
            ]
            connectors = connectors.loc[
                np.logical_and(connectors[d] > mi, connectors[d] < ma)
            ]

        # eliminate edges if any of their nodes are not in the ROI
        tnids = set(treenodes["treenode_id"])
        cids = set(connectors["connector_id"])
        edges = self.edges.loc[
            np.logical_and(
                self.edges["connector_id"].isin(cids),
                np.logical_and(
                    self.edges["treenode_id_pre"].isin(tnids),
                    self.edges["treenode_id_post"].isin(tnids),
                ),
            )
        ]

        # eliminate tree&connector nodes if they are no longer involved in an edge
        # e.g. its partner was outside the ROI
        tnids = set(edges["treenode_id_pre"]).union(edges["treenode_id_post"])
        treenodes = treenodes.loc[treenodes["treenode_id"].isin(tnids)]
        cids = set(edges["connector_id"])
        connectors = connectors.loc[connectors["connector_id"].isin(cids)]

        # eliminate skeletons if none of their treenodes remain
        skids = set(treenodes["skeleton_id"])
        skeletons = self.skeletons.loc[self.skeletons["skeleton_id"].isin(skids)]

        out = type(self)(edges, treenodes, connectors, skeletons)
        if copy:
            out = out.copy(True)

        return out

    @classmethod
    def from_nodes(
        cls,
        connector_ids: Sequence[int],
        post_treenode_ids: Sequence[int],
        remote_instance=None,
    ) -> Self:
        """Get edge information for given connector node -> postsynaptic node pairs.

        This is the minimum information needed to uniquely identify
        synapses from one cell to another.

        Synapse ``i`` is from ``connector_ids[i]`` to ``post_treenode_ids[i]``.

        Parameters
        ----------
        connector_ids : Sequence[int]
        post_treenode_ids : Sequence[int]
        remote_instance : pymaid.CatmaidInstance, optional
            CATMAID API accessor, by default None (global)
        """
        logger.info("Fetching details of %s connectors", len(connector_ids))

        edge_builder = DfBuilder(
            subset_dict(
                edge_columns,
                [
                    "treenode_id_pre",
                    "connector_id",
                    "treenode_id_post",
                ],
            )
        )
        edge_df = pymaid.get_connector_details(
            connector_ids, remote_instance=remote_instance
        )
        wanted = set(zip(connector_ids, post_treenode_ids))
        node_to_skid = dict()
        for row in edge_df.itertuples(index=False):
            for post_tnid in row.postsynaptic_to_node:
                if (row.connector_id, post_tnid) in wanted:
                    edge_builder.append(
                        [
                            row.presynaptic_to_node,
                            row.connector_id,
                            post_tnid,
                        ]
                    )
                    node_to_skid[row.presynaptic_to_node] = row.presynaptic_to

        edge_df = edge_builder.build()
        logger.info("Getting skeleton ID for %s nodes", len(edge_df))

        for k, v in pymaid.get_skid_from_node(
            list({n for n in edge_df["treenode_id_post"] if n not in node_to_skid}),
            remote_instance=remote_instance,
        ).items():
            node_to_skid[int(k)] = int(v)

        edge_df.sort_values(
            ["treenode_id_pre", "connector_id", "treenode_id_post"],
            ignore_index=True,
            inplace=True,
        )
        edge_df.insert(0, "edge_id", np.arange(len(edge_df), dtype="uint64"))

        logger.info("Fetching locations of %s nodes", len(node_to_skid))
        treenode_df = pymaid.get_node_location(
            list(node_to_skid), remote_instance=remote_instance
        )
        treenode_df.insert(
            1, "skeleton_id", [node_to_skid[tn] for tn in treenode_df["node_id"]]
        )
        treenode_df.rename(columns={"node_id": "treenode_id"}, inplace=True)
        treenode_df.sort_values(
            ["skeleton_id", "treenode_id"], inplace=True, ignore_index=True
        )

        conn_ids = edge_df["connector_id"].unique()
        logger.info("Fetching locations of %s connectors", len(conn_ids))

        connector_df = pymaid.get_node_location(
            conn_ids.tolist(), remote_instance=remote_instance
        )
        connector_df.rename(columns={"node_id": "connector_id"}, inplace=True)
        connector_df.sort_values("connector_id", inplace=True, ignore_index=True)

        skids = sorted(treenode_df["skeleton_id"].unique())
        logger.info("Fetching names of %s skeletons", len(skids))
        names_dict = pymaid.get_names(skids, remote_instance=remote_instance)
        names = [names_dict[str(skid)] for skid in skids]
        skel_df = pd.DataFrame.from_dict(
            {"skeleton_id": skids, "neuron_name": names}
        ).astype(skeleton_columns)

        return cls(edge_df, treenode_df, connector_df, skel_df)

    def translate(self, translation: dict[Dim, float], copy=False) -> Self:
        if copy:
            out = self.copy(True)
        else:
            out = self

        keys = list(translation)
        values = [translation[d] for d in keys]
        out.connectors[keys] += values
        out.treenodes[keys] += values
        return out

    def copy(self, deep=False):
        if deep:
            return deepcopy(self)
        else:
            return copy(self)

    @classmethod
    def empty(cls) -> Self:
        return cls(
            DfBuilder(edge_columns).build(),
            DfBuilder(treenode_columns).build(),
            DfBuilder(connector_columns).build(),
            DfBuilder(skeleton_columns).build(),
        )

    @classmethod
    def _from_skeletons_pre_post(
        cls, pre_skids: Sequence[int], post_skids: Sequence[int], remote_instance=None
    ) -> Self:
        df = pymaid.get_connectors_between(
            pre_skids, post_skids, remote_instance=remote_instance
        )
        return cls.from_nodes(df["connector_id"], df["node2_id"])

    @classmethod
    def _from_skeletons_pre(
        cls, pre_skids: Sequence[int], remote_instance=None
    ) -> Self:
        df = pymaid.get_partners(
            pre_skids,
            directions=["outgoing"],
            min_size=1,
            remote_instance=remote_instance,
        )
        df.drop(
            columns=["neuron_name", "skeleton_id", "num_nodes", "relation", "total"],
            inplace=True,
        )

        post_skids = []
        for skid in df.columns:
            if df[skid].sum() >= 1:
                post_skids.append(int(skid))
        return cls._from_skeletons_pre_post(pre_skids, post_skids, remote_instance)

    @classmethod
    def _from_skeletons_post(
        cls, post_skids: Sequence[int], remote_instance=None
    ) -> Self:
        df = pymaid.get_connector_links(
            post_skids, chunk_size=50, remote_instance=remote_instance
        )
        df = df[df["relation"] == "postsynaptic_to"]
        return cls.from_nodes(df["connector_id"], df["node_id"])

    @classmethod
    def from_skeletons(
        cls,
        pre_skeleton_ids: Optional[Sequence[int]] = None,
        post_skeleton_ids: Optional[Sequence[int]] = None,
        remote_instance=None,
    ) -> Self:
        """Get all connectors from any of the pre- to any of the post-skeletons.

        If pre_ is None, use all upstream partners of the post_.
        If post_ is None, use all downstream partners of the pre_.
        If both are None, returns an empty set of connectors.

        Parameters
        ----------
        pre_skeleton_ids : Optional[Sequence[int]], optional
            Presynaptic skeletons, by default None
            (use all upstream partners of postsynaptic skeletons).
        post_skeleton_ids : Optional[Sequence[int]], optional
            Postsynaptic skeletons, by default None
            (use all downstream partners of presynaptic skeletons).
        remote_instance : pymaid.CatmaidInstance, optional
            CATMAID API accessor, by default None (global instance)
        """
        if pre_skeleton_ids is None and post_skeleton_ids is None:
            raise ValueError("Neither pre nor post skeletons given")

        if (pre_skeleton_ids is not None and len(pre_skeleton_ids) == 0) or (
            post_skeleton_ids is not None and len(post_skeleton_ids) == 0
        ):
            return cls.empty()

        if pre_skeleton_ids is None:
            return cls._from_skeletons_post(post_skeleton_ids, remote_instance)

        if post_skeleton_ids is None:
            return cls._from_skeletons_pre(pre_skeleton_ids, remote_instance)

        return cls._from_skeletons_pre_post(
            pre_skeleton_ids, post_skeleton_ids, remote_instance
        )
