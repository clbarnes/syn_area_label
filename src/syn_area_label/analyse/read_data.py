from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Type
import numpy as np

from cremi import CremiFile
import pandas as pd

from .area import DEFAULT_AREA_CALCULATOR, AreaCalculator

from ..synapses import EdgeTables
from ..utils import Dim, TABLES_GROUP

# volume
# resolution
# dimension names
# edge tables
# pre ID -> label
logger = logging.getLogger(__name__)

LABEL_VOLUME = "/volumes/labels/canvas"


class AreaInfo:
    def __init__(
        self,
        volume: np.ndarray,
        resolution: tuple[float, float, float],
        dimensions: tuple[Dim, Dim, Dim],
        edge_tables: EdgeTables,
        label_ids: dict[int, int],
    ) -> None:
        """Internal constructor.
        Use from_cremi alternate constructor instead.

        Parameters
        ----------
        volume : np.ndarray
        resolution : tuple[float, float, float]
        dimensions : tuple[Dim, Dim, Dim]
        edge_tables : EdgeTables
        label_ids : dict[int, int]
            Dict from presynaptic location ID to the label ID used
            for the pre-post edge.
        """
        self.volume = volume
        self.resolution = resolution
        self.dimensions = dimensions
        self.edge_tables = edge_tables
        self.label_ids = label_ids

    @classmethod
    def from_cremi(cls, fpath: Path, label_ids: Optional[dict[int, int]] = None):
        """Create from a CREMI file.

        Parameters
        ----------
        fpath : Path
            Path to the cremi file
        label_ids : Optional[dict[int, int]], optional
            Map from edge ID (a small integer) to the label ID used in the painted volume.
            If None (default), tries to determine from annotations in the CREMI file.
        """
        # todo: get site_ids?
        etables = EdgeTables.from_hdf5(fpath, TABLES_GROUP)
        cf = CremiFile(fpath, "r")

        if label_ids is None:
            anns = cf.read_annotations()
            # todo: check key is correct type
            label_ids = dict()
            for eid in etables.edges["edge_id"]:
                val = anns.comments[eid].strip()
                if not val:
                    logger.warning("No label found for edge %s", eid)
                    continue
                label_ids[int(eid)] = int(val)

        clefts = cf.read_clefts()
        res = tuple(clefts.resolution)
        dims = tuple(cf.h5file.attrs["dimensions"])

        logger.debug("Reading painted data")
        vol = cf.h5file[LABEL_VOLUME][:]

        return cls(vol, res, dims, etables, label_ids)

    def calculate_area(
        self, calculator: Type[AreaCalculator] = DEFAULT_AREA_CALCULATOR
    ) -> pd.DataFrame:
        """Calculate the synaptic areas in this volume.

        Parameters
        ----------
        calculator : Type[AreaCalculator]
            Python class (NOT instance of the object) used to calculate the area.

        Returns
        -------
        pd.DataFrame
            pre_skeleton_id, pre_treenode_id, connector_id, post_treenode_id, post_skeleton_id, area
        """
        # todo: avoid hard coding slicing dim?
        calc = calculator(
            self.volume, self.resolution, self.dimensions, self.dimensions[0]
        )
        dtypes = {
            "pre_skeleton_id": "uint64",
            "pre_treenode_id": "uint64",
            "connector_id": "uint64",
            "post_treenode_id": "uint64",
            "post_skeleton_id": "uint64",
            "area": "float64",
        }
        tn_to_skel = dict(
            zip(
                self.edge_tables.treenodes["treenode_id"],
                self.edge_tables.treenodes["skeleton_id"],
            )
        )
        edge_to_row = {
            row.edge_id: row for row in self.edge_tables.edges.itertuples(index=False)
        }
        label_ids = {k: v for k, v in self.label_ids.items() if k in edge_to_row}

        areas = calc.calculate(list(label_ids.values()))
        if len(self.label_ids) != len(edge_to_row):
            logger.warning(
                "Calculating area for %s annotated labels, but there are %s edges",
                len(label_ids),
                len(edge_to_row),
            )

        rev_labels = {v: k for k, v in label_ids.items()}
        rows = []
        for label_id, area in areas.items():
            edge_id = rev_labels[label_id]
            einfo = edge_to_row[edge_id]
            rows.append(
                [
                    tn_to_skel[einfo.treenode_id_pre],
                    einfo.treenode_id_pre,
                    einfo.connector_id,
                    einfo.treenode_id_post,
                    tn_to_skel[einfo.treenode_id_post],
                    area,
                ]
            )
        return pd.DataFrame(rows, columns=list(dtypes)).astype(dtypes)
