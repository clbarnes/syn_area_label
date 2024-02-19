from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from skimage.morphology import skeletonize
from scipy.ndimage import convolve

from ..utils import Dim

from typing import Iterator, Optional

import networkx as nx


class AreaCalculator(ABC):
    """Abstract base class for things which can calculate a synapse's area."""

    def __init__(
        self,
        volume: np.ndarray,
        resolution: tuple[float, float, float],
        dimensions: tuple[Dim, Dim, Dim],
        slicing_dim: Dim,
    ) -> None:
        self.volume = volume
        self.resolution = resolution
        self.dimensions = dimensions
        self.slicing_dim = slicing_dim

    def _slicing_dim_idx(self):
        dim_idxs = [
            idx for idx, dim in enumerate(self.dimensions) if dim == self.slicing_dim
        ]
        if len(dim_idxs) != 1:
            raise RuntimeError(
                f"Expected 1 dimension called {self.slicing_dim}, got {len(dim_idxs)}"
            )
        return dim_idxs.pop()

    def _px_area(self):
        """XY diagonal * Z"""
        slidx = self._slicing_dim_idx()
        diag_sq = sum(r**2 for (idx, r) in enumerate(self.resolution) if idx != slidx)
        return self.resolution[slidx] * diag_sq

    def _iter_slices(self):
        sl_idx = self._slicing_dim_idx()
        vol = np.moveaxis(self.volume, sl_idx, 0)
        for layer in vol:
            yield layer

    def _iter_mask(self, label_id: int):
        for sl in self._iter_slices():
            yield sl == label_id

    @abstractmethod
    def calculate(self, label_ids: list[int]) -> dict[int, float]:
        return {lb: 0 for lb in label_ids}


class SliceCount(AreaCalculator):
    """How many slices a synapse appears on."""

    def calculate(self, label_ids: list[int]) -> dict[int, float]:
        out = super().calculate(label_ids)

        for sl in self._iter_slices():
            for label_id in label_ids:
                if np.any(sl == label_id):
                    out[label_id] += 1

        return out


class SkeletonizingAreaCalculator(AreaCalculator, ABC):
    def _iter_mask(self, label_id: int):
        for sl in super()._iter_mask(label_id):
            yield skeletonize(sl)


class PixelCount(SkeletonizingAreaCalculator):
    """Skeletonise all labels and count pixels"""

    def calculate(self, label_ids: list[int]) -> dict[int, float]:
        out = super().calculate(label_ids)
        for sl in self._iter_slices():
            for label_id in label_ids:
                out[label_id] += (sl == label_id).sum()
        return out


class PixelCountTimesZ(PixelCount):
    """Skeletonise labels, count pixels, multiply by XY diagonal * Z"""

    def calculate(self, label_ids: list[int]) -> dict[int, float]:
        counts = super().calculate(label_ids)
        px_ar = self._px_area()
        return {k: v * px_ar for k, v in counts.items()}


class LinearAreaCalculator(SkeletonizingAreaCalculator):
    """
    Skeletonize 2D planes in 3D images, convolve to find actual count of diagonal and on-face edges.
    """

    def __init__(
        self,
        volume: np.ndarray,
        resolution: tuple[float, float, float],
        dimensions: tuple[Dim, Dim, Dim],
        slicing_dim: Dim,
    ) -> None:
        super().__init__(volume, resolution, dimensions, slicing_dim)
        # self.origin = (0, 0)
        self.origin = 0
        self.kernel = self._kernel()

    def _kernel(self):
        plane_res = [
            res
            for dim, res in zip(self.dimensions, self.resolution)
            if dim != self.slicing_dim
        ]

        diag_len = np.linalg.norm(plane_res)
        return (
            np.array(
                [
                    [diag_len, plane_res[0], diag_len],
                    [plane_res[1], 0, plane_res[1]],
                    [diag_len, plane_res[0], diag_len],
                ]
            )
            # Divide by 2 because each edge will be represented twice
            / 2
        )

    def length(self, skeletonized_2d):
        return convolve(
            skeletonized_2d.astype(float),
            self.kernel,
            mode="constant",
            cval=0,
            origin=self.origin,
        )[skeletonized_2d].sum()

    def calculate(self, label_ids: list[int]) -> dict[int, float]:
        slice_res = self.resolution[self._slicing_dim_idx()]
        out = super().calculate(label_ids)
        for label in label_ids:
            for z_plane in self._iter_mask(label):
                out[label] += self.length(z_plane) * slice_res
        return out


DEFAULT_AREA_CALCULATOR = LinearAreaCalculator
