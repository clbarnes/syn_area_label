import logging
from pathlib import Path
from typing import Literal, NamedTuple, Optional
from collections.abc import Sequence

import pymaid
import numpy as np
import pandas as pd

from .utils import DfBuilder

logger = logging.getLogger(__name__)

edge_columns = {
    "edge_id": np.uint64,
    "pre_skeleton_id": np.uint64,
    "pre_treenode_id": np.uint64,
    "connector_id": np.uint64,
    "post_treenode_id": np.uint64,
    "post_skeleton_id": np.uint64,
}
treenode_columns = {
    "treenode_id": np.uint64,
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
skeleton_columns = {"skeleton_id": np.uint64, "name": str}


def write_tsv(path: Path, df: pd.DataFrame):
    return df.to_csv(path, sep="\t", index=False)


def read_tsv(path: Path, dtype=None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=dtype)


Dim = Literal["x", "y", "z"]


def subset_dict(d, keys) -> dict:
    return {k: d[k] for k in keys}


class EdgeTables(NamedTuple):
    edges: pd.DataFrame
    treenodes: pd.DataFrame
    connectors: pd.DataFrame
    skeletons: pd.DataFrame

    def to_directory(self, path: Path, padding: Optional[dict[Dim, float]] = None):
        path.mkdir(exist_ok=True, parents=True)
        for name, df in self._asdict().items():
            p = path / f"{name}.tsv"
            write_tsv(p, df)

        if padding is None:
            return

        df = self.rois(padding)
        write_tsv(path / "rois.tsv", df)

    @classmethod
    def from_directory(cls, path: Path):
        return cls(
            read_tsv(path / "edges.tsv", edge_columns),
            read_tsv(path / "treenodes.tsv", treenode_columns),
            read_tsv(path / "connectors.tsv", connector_columns),
            read_tsv(path / "skeletons.tsv", skeleton_columns),
        )

    def merge_locations(self) -> pd.DataFrame:
        df = self.edges
        df = pd.merge(df, self.connectors, on="connector_id", suffixes=(None, "_conn"))
        df = pd.merge(df, self.treenodes, on="pre_treenode_id", suffixes=(None, "_pre"))
        return pd.merge(
            df, self.treenodes, on="post_treenode_id", suffixes=(None, "_post")
        )

    def rois(self, padding: dict[Dim, float]) -> pd.DataFrame:
        merged = self.merge_locations()
        suffixes = ["_conn", "_pre", "_post"]
        new_cols = dict()
        axis = 1  # todo: check axis
        for d in ["x", "y", "z"]:
            xyz = merged[[f"{d}{suff}" for suff in suffixes]]
            new_cols[f"{d}_min"] = xyz.min(axis=axis) - padding[d]
            new_cols[f"{d}_max"] = xyz.max(axis=axis) + padding[d]

        return pd.DataFrame.from_dict({"edge_id": merged["edge_id"], **new_cols})

    @classmethod
    def from_nodes(
        cls,
        connector_ids: Sequence[int],
        post_treenode_ids: Sequence[int],
        remote_instance=None,
    ):
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
                    "pre_skeleton_id",
                    "pre_treenode_id",
                    "connector_id",
                    "post_treenode_id",
                ],
            )
        )
        edge_df = pymaid.get_connector_details(
            connector_ids, remote_instance=remote_instance
        )
        wanted = set(zip(connector_ids, post_treenode_ids))
        for row in edge_df.itertuples(index=False):
            for post_tnid in row.postsynaptic_to_node:
                if (row.connector_id, post_tnid) in wanted:
                    edge_builder.append(
                        [
                            row.presynaptic_to,
                            row.presynaptic_to_node,
                            row.connector_id,
                            post_tnid,
                        ]
                    )

        edge_df = edge_builder.build()
        logger.info("Getting skeleton ID for %s nodes", len(edge_df))
        node_to_skid = {
            int(k): int(v)
            for k, v in pymaid.get_skid_from_node(
                edge_df["post_treenode_id"], remote_instance=remote_instance
            ).items()
        }
        edge_df["post_skeleton_id"] = np.array(
            [node_to_skid[n] for n in edge_df["post_treenode_id"]],
            edge_columns["post_skeleton_id"],
        )
        edge_df.sort_values(
            ["pre_skeleton_id", "post_skeleton_id", "connector_id", "post_treenode_id"],
            ignore_index=True,
            inplace=True,
        )
        edge_df.insert(0, "edge_id", np.arange(len(edge_df), dtype="uint64"))

        tnids = list(set(edge_df["pre_treenode_id"]).union(edge_df["post_treenode_id"]))
        logger.info("Fetching locations of %s nodes", len(tnids))
        treenode_df = pymaid.get_node_location(tnids, remote_instance=remote_instance)
        treenode_df.rename(columns={"node_id": "treenode_id"}, inplace=True)
        treenode_df.sort_values("treenode_id", inplace=True, ignore_index=True)

        conn_ids = edge_df["connector_id"].unique()
        logger.info("Fetching locations of %s connectors", len(conn_ids))

        connector_df = pymaid.get_node_location(
            conn_ids.tolist(), remote_instance=remote_instance
        )
        connector_df.rename(columns={"node_id": "connector_id"}, inplace=True)
        connector_df.sort_values("connector_id", inplace=True, ignore_index=True)

        skids = sorted(
            set(edge_df["pre_skeleton_id"]).union(edge_df["post_skeleton_id"])
        )
        logger.info("Fetching names of %s skeletons", len(skids))
        names_dict = pymaid.get_names(skids, remote_instance=remote_instance)
        names = [names_dict[str(skid)] for skid in skids]
        skel_df = pd.DataFrame.from_dict({"skeleton_id": skids, "name": names}).astype(
            skeleton_columns
        )

        return cls(edge_df, treenode_df, connector_df, skel_df)

    @classmethod
    def empty(cls):
        return cls(
            DfBuilder(edge_columns).build(),
            DfBuilder(treenode_columns).build(),
            DfBuilder(connector_columns).build(),
            DfBuilder(skeleton_columns).build(),
        )

    @classmethod
    def _from_skeletons_pre_post(
        cls, pre_skids: Sequence[int], post_skids: Sequence[int], remote_instance=None
    ):
        df = pymaid.get_connectors_between(
            pre_skids, post_skids, remote_instance=remote_instance
        )
        return cls.from_nodes(df["connector_id"], df["node2_id"])

    @classmethod
    def _from_skeletons_pre(cls, pre_skids: Sequence[int], remote_instance=None):
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
    def _from_skeletons_post(cls, post_skids: Sequence[int], remote_instance=None):
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
    ):
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
            return cls._from_skeletons_pre_post(
                pre_skeleton_ids, post_skeleton_ids, remote_instance
            )

        return cls._from_skeletons_pre(pre_skeleton_ids, remote_instance)
