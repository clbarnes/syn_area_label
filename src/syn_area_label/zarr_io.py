from typing import Any, Iterable, Optional
import logging

import h5py
import numpy as np
import zarr
from dask import array as da
from h5py import File
from tqdm import tqdm

from .utils import VoxelCoord, WorldCoord

logger = logging.getLogger(__name__)


def n5_to_dask(uri: str, dataset: str) -> da.Array:
    logger.debug("Opening n5 dataset")
    store = zarr.N5FSStore(uri, mode="r")
    group = zarr.open_group(store, "r")
    ds = group[dataset]
    return da.from_zarr(ds)


class WrappedVolume:
    def __init__(
        self, array: da.Array, translation: WorldCoord, resolution: WorldCoord
    ) -> None:
        """Assumes dask array in ZYX"""
        self.array = array
        self.translation = translation
        self.resolution = resolution

    def get_roi(
        self, offset: WorldCoord, shape: WorldCoord
    ) -> tuple[WorldCoord, da.Array]:
        logger.debug("Getting ROI at %s", offset)
        offset_vox = offset.to_voxel(self.resolution, self.translation, np.floor)
        end = WorldCoord(*(offset.to_ndarray() + shape.to_ndarray()))
        end_vox = end.to_voxel(self.resolution, self.translation, np.ceil)
        slicing = tuple(slice(od, ed) for od, ed in zip(offset_vox, end_vox))
        # may need to call compute here
        sub_arr = self.array[slicing]

        actual_offset = offset_vox.to_world(self.resolution, self.translation)
        return actual_offset, sub_arr


class RoiExtractor:
    def __init__(
        self,
        raw_volume: WrappedVolume,
        subvol_offset: VoxelCoord,
        subvol_shape: VoxelCoord,
        hdf5_file: File,
        ds_kwargs: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        self.raw_volume = raw_volume
        self.subvol_offset = subvol_offset
        self.subvol_shape = subvol_shape
        self.hdf5_file = hdf5_file
        self.ds_kwargs = ds_kwargs or dict()
        self.attributes = attributes or dict()

        res = self.raw_volume.resolution.to_ndarray()
        self.translation = WorldCoord(
            *(self.raw_volume.translation.to_ndarray() + res * self.subvol_offset)
        )
        self.resolution = self.raw_volume.resolution

    def _create_dataset(self) -> h5py.Dataset:
        g = self.hdf5_file.require_group("/volumes")
        self.ds_kwargs.setdefault("compression", "lzf")
        self.ds_kwargs.setdefault(
            "chunks", tuple(c[0] for c in self.raw_volume.array.chunks)
        )
        ds = g.create_dataset(
            "raw",
            shape=self.subvol_shape.to_ndarray(),
            dtype=self.raw_volume.array.dtype,
            **self.ds_kwargs
        )
        a = ds.attrs
        a.update(self.attributes)
        a["resolution"] = self.resolution.to_ndarray()
        a["offset_from_project"] = self.translation.to_ndarray()
        return ds

    def dataset(self) -> h5py.Dataset:
        try:
            ds: h5py.Dataset = self.hdf5_file["/volumes/raw"]
        except KeyError:
            ds = self._create_dataset()
        return ds

    def extract_roi(
        self,
        offset: WorldCoord,
        shape: WorldCoord,
        dataset: Optional[h5py.Dataset] = None,
    ) -> h5py.Dataset:
        if dataset is None:
            dataset = self.dataset()

        actual_offset, roi = self.raw_volume.get_roi(offset, shape)
        internal_offset = actual_offset.to_voxel(self.resolution, self.translation)
        slicing = tuple(
            slice(int(i), int(i + s)) for i, s in zip(internal_offset, roi.shape)
        )
        logger.debug("Writing ROI to %s", slicing)
        dataset[slicing] = roi
        return dataset

    def extract_roi_around_point(self, point: WorldCoord, padding: float, dataset=None):
        shape = WorldCoord(padding * 2, padding * 2, padding * 2)
        offset = point.to_ndarray() - [padding] * 3
        return self.extract_roi(WorldCoord(*offset), shape, dataset)

    def extract_rois_around_points(
        self, points: Iterable[WorldCoord], padding: float, dataset=None
    ):
        for p in tqdm(points, description="Extracting ROIs"):
            dataset = self.extract_roi_around_point(p, padding, dataset)
