#!/usr/bin/env python3
"""
Write what this script is for, and how to use it, here.
"""
import csv
import logging

from syn_area_label.constants import DATA_DIR
from syn_area_label.utils import setup_logging

logger = logging.getLogger()


def read_csv(input_path):
    """Read a list of dicts from a TSV file."""
    logger.debug("Reading TSV at %s", input_path)

    with open(input_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def write_csv(data, output_path):
    """Write a TSV file from a list of dicts."""
    logger.debug("Writing %s records to TSV at %s", len(data), output_path)

    with open(output_path, "w") as f:
        writer = csv.DictWriter(f, list(data[0]), delimiter="\t")
        writer.writerows(data)


def main():
    """This is what your script should be built around."""
    data = read_csv(DATA_DIR / "input" / "example.tsv")
    write_csv(data, DATA_DIR / "output" / "example.tsv")


# ensures that the code is only run when this file is called as a script
if __name__ == "__main__":
    # using logging, rather than print statements,
    # gives you access to information from underlying libraries;
    # and you can leave the log lines there and just change the level
    # if you don't need to see them any more
    setup_logging()
    logger.info("Starting script")

    main()

    logger.info("Finished script")
