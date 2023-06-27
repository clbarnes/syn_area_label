"""Implementation of bigCAT HDF5 format v0.2.

Documented here: https://github.com/saalfeldlab/bigcat/wiki/HDF5-Schema

Alternative implementation here: https://github.com/cremi/cremi_python
"""
from __future__ import annotations

import logging
from abc import ABC
from collections.abc import Mapping, Sequence
from typing import Any, NewType, Optional, Union

import h5py
import numpy as np
import pandas as pd
from h5py import File

from syn_area_label.utils import WorldCoord

from .constants import DIMS

logger = logging.getLogger(__name__)


def require_parent_groups(hdf5_file: File, ds_path: str):
    *parents, _ = ds_path.split("/")
    name = ""
    for p in parents:
        name = f"{name}/{p}"
        hdf5_file.require_group(name)


class Volume(ABC):
    dtype: np.dtype = None

    def __init__(
        self,
        data: np.ndarray,
        resolution=WorldCoord(1, 1, 1),
        offset_from_project=WorldCoord(0, 0, 0),
        comment="",
    ) -> None:
        if self.dtype is not None and data.dtype != self.dtype:
            raise ValueError(f"Needed dtype {self.dtype}, got {data.dtype}")
        self.data = data
        self.resolution = resolution
        self.offset_from_project = offset_from_project
        self.comment = comment

    def _to_hdf5(
        self,
        hdf5_file: File,
        ds_path: str,
        ds_kwargs: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
    ):
        if ds_kwargs is None:
            ds_kwargs = dict()

        require_parent_groups(hdf5_file, ds_path)

        ds = hdf5_file.create_dataset(ds_path, data=self.data, **ds_kwargs)
        a = ds.attrs
        if attributes:
            a.update(attributes)
        a["resolution"] = self.resolution.to_ndarray()
        a["offset_from_project"] = self.offset_from_project.to_ndarray()
        a["comment"] = self.comment

    @classmethod
    def _from_hdf5(cls, hdf5_file: File, ds_path: str):
        ds = hdf5_file[ds_path]
        a = ds.attrs
        return cls(
            ds[:],
            WorldCoord(*a["resolution"]),
            WorldCoord(*a["offset_from_project"]),
            a["comment"],
        )

    def world_shape(self):
        return self.resolution.to_ndarray() * self.data.shape


class RawVolume(Volume):
    dtype = np.dtype(np.uint8)
    name = "/volumes/raw"

    def to_hdf5(
        self,
        hdf5_file: File,
        ds_kwargs: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
    ):
        return self._to_hdf5(hdf5_file, self.name, ds_kwargs, attributes)

    @classmethod
    def from_hdf5(cls, hdf5_file: File):
        return cls._from_hdf5(hdf5_file, cls.name)


class LabelVolume(Volume):
    dtype = np.dtype(np.uint64)
    group = "/volumes/labels"

    def __init__(
        self,
        name: str,
        data: np.ndarray,
        resolution=WorldCoord(1, 1, 1),
        offset_from_project=WorldCoord(0, 0, 0),
        offset_from_raw=WorldCoord(0, 0, 0),
        comment="",
    ) -> None:
        super().__init__(data, resolution, offset_from_project, comment)
        self.offset_from_raw = offset_from_raw
        self.name = f"/volumes/labels/{name}"

    def to_hdf5(
        self,
        hdf5_file: File,
        ds_kwargs: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
    ):
        attributes = attributes or dict()
        attributes["offset"] = self.offset_from_raw
        return self._to_hdf5(
            hdf5_file, f"{self.group}/{self.name}", ds_kwargs, attributes
        )

    @classmethod
    def from_hdf5(cls, hdf5_file: File, name: str):
        path = f"{cls.group}/{name}"
        p = super()._from_hdf5(hdf5_file, path)
        offset_arr = hdf5_file[path].attrs.get("offset", [0, 0, 0])
        return cls(
            name,
            p.data,
            p.resolution,
            p.offset_from_project,
            WorldCoord(*offset_arr),
            p.comment,
        )


class Dataset:
    def __init__(self, array: np.ndarray, attributes: dict[str, Any] = None) -> None:
        self.array = array
        self.attributes = attributes or dict()

    def to_hdf5(
        self,
        hdf5_file: File,
        name: str,
        ds_kwargs: Optional[dict[str, Any]] = None,
    ):
        if ds_kwargs is None:
            ds_kwargs = dict()
        require_parent_groups(hdf5_file, name)

        ds = hdf5_file.create_dataset(name, data=self.array)
        ds.attrs.update(self.attributes)

    @classmethod
    def from_hdf5(cls, hdf5_file: File, name: str):
        ds = hdf5_file[name]
        return cls(ds[:], dict(ds.attrs))


AnnotationType = NewType("AnnotationType", str)

ann_dtypes = {
    "id": np.uint64,
    "type": str,
    "z": np.float64,
    "y": np.float64,
    "x": np.float64,
}


class AnnotationTableBuilder:
    def __init__(self) -> None:
        self.rows = []

    def add_row(self, row: Union[Sequence, Mapping]):
        if isinstance(row, Sequence):
            self.rows.append(row)
        elif isinstance(row, Mapping):
            self.rows.append([row[k] for k in ann_dtypes])
        else:
            raise ValueError("row should be a sequence or a mapping")

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows, columns=list(ann_dtypes), dtype=object)
        return df.astype(ann_dtypes)


class Annotations:
    def __init__(
        self,
        ann_table: Optional[pd.DataFrame] = None,
        comments: Optional[dict[int, str]] = None,
        special: Optional[dict[AnnotationType, dict[str, Dataset]]] = None,
    ):
        self.ann_table = self._verify_anns(ann_table)
        self.comments = comments or dict()
        self.special = special or dict()

        self.max_id: int = 0 if len(self.ann_table) == 0 else self.ann_table["id"].max()

    @classmethod
    def from_partners(
        cls,
        locs: dict[int, WorldCoord],
        pre_post: dict[int, int],
    ):
        """Generate annotations from pre-post partners."""
        visited_postsyn = set()
        ann_builder = AnnotationTableBuilder()
        partners = []
        for presyn_id, postsyn_id in pre_post.items():
            presyn_loc = locs[presyn_id]
            postsyn_loc = locs[postsyn_id]
            ann_builder.add_row([presyn_id, "presynaptic_site", *presyn_loc])
            partners.append([presyn_id, postsyn_id])
            if postsyn_id not in visited_postsyn:
                ann_builder.add_row([postsyn_id, "postsynaptic_site", *postsyn_loc])
                visited_postsyn.add(postsyn_id)

        ann_table = ann_builder.to_dataframe()
        special = {
            "presynaptic_site": {"partners": Dataset(np.array(partners, np.uint64))}
        }
        return cls(ann_table, special=special)

    def _verify_anns(self, df: Optional[pd.DataFrame]):
        if df is None:
            df = pd.DataFrame([], columns=list(ann_dtypes)).astype(ann_dtypes)
        elif set(df.columns) != set(ann_dtypes):
            raise ValueError("Annotation table has incorrect columns")
        return df

    def _next_id(self):
        self.max_id += 1
        return self.max_id

    def set_annotation(
        self, atype: AnnotationType, location: WorldCoord, aid: Optional[int] = None
    ):
        if aid is None:
            aid = self._next_id()

        loc_dict = location._asdict()

        logical = self.ann_table["id"] == aid
        if not np.any(logical):
            row = {"id": aid, "type": atype, **loc_dict}
            self.ann_table.append(row)
        else:
            row = self.ann_table[logical]
            row["type"] = atype

            for d in DIMS:
                row[d] = loc_dict[d]
        return aid

    def to_hdf5(self, hdf5_file: File):
        logger.info("Writing annotations")
        g = hdf5_file.require_group("/annotations")
        g.create_dataset("ids", data=self.ann_table["id"], dtype=np.uint64)
        g.create_dataset(
            "types", data=self.ann_table["type"], dtype=h5py.string_dtype()
        )
        g.create_dataset(
            "locations", data=self.ann_table[list(DIMS)].to_numpy(), dtype=np.float64
        )
        g_comm = g.require_group("comments")
        target_ids = []
        comments = []
        for k, v in self.comments.items():
            target_ids.append(k)
            comments.append(v)
        g_comm.create_dataset("target_ids", data=target_ids, dtype=np.uint64)
        g_comm.create_dataset("comments", data=comments, dtype=h5py.string_dtype())

        for atype, datasets in self.special.items():
            g_atype = g.require_group(atype)
            for name, ds in datasets.items():
                ds_h = g_atype.create_dataset(name, data=ds.array)
                ds_h.attrs.update(ds.attributes)

    @classmethod
    def from_hdf5(cls, hdf5_file: File):
        g_ann = hdf5_file["/annotations"]
        ann_cols = {
            "id": g_ann["ids"],
            "type": g_ann["types"],
        }
        locs = g_ann["locations"]
        for idx, d in enumerate(DIMS):
            ann_cols[d] = locs[:, idx]
        ann_table = pd.DataFrame(ann_cols)
        g_comm = g_ann["comments"]
        comments = dict(zip(g_comm["target_ids"], g_comm["comments"]))

        reserved = {"id", "type", "locations", "comments"}
        special = {}

        for atype in g_ann.keys():
            if atype in reserved:
                continue
            data = special.setdefault(atype, dict())
            for ds_name in g_ann[atype]:
                ds = g_ann[ds_name]
                data[ds_name] = Dataset(ds[:], dict(ds.attrs))

        return cls(ann_table, comments, special)
