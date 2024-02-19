#!/usr/bin/env python3
from pathlib import Path
import logging

from syn_area_label.analyse.read_data import AreaInfo

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

out_dir = Path(__file__).resolve().parent.parent.joinpath("data/output/large/examples")

# See the warning in the `painting_instructions.md`
# These were originally store in the CREMI file but there is a bug in bigCAT preventing this
label_ids = {3: 14320177}

ainfo = AreaInfo.from_cremi(out_dir.joinpath("rois/conn_3819413.hdf5"), label_ids)
area_table = ainfo.calculate_area()

print(area_table)
