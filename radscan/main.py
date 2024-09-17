#!/usr/bin/env python

# import os
import sys
import logging

# Import necessary modules from the RadScan package
# from radscan import RSImage, ROI
# from radscan import Calibration # only needed if we want to plot the calibration curve
# from radscan.workflow import analyze_simple_image, analyze_simple_roi


logger = logging.getLogger(__name__)


def main(args=None):
    """
    Entry point for the RadScan tool.
    Users can extend this script to add more functionality as needed.
    """

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting RadScan tool...")

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
