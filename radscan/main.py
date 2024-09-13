#!/usr/bin/env python

import os
import sys
import logging

# Import necessary modules from the RadScan package
from radscan.calibration import Calibration
from radscan.image import RSImage


logger = logging.getLogger(__name__)


def main(args=None):
    """
    Entry point for the RadScan tool.
    Users can extend this script to add more functionality as needed.
    """

    # Load example resources
    resources_dir = os.path.join(os.path.dirname(__file__), '../resources')
    example_tiff = os.path.join(resources_dir, 'example_scan.tif')

    print(resources_dir)

    # Load the TIFF image
    # image = RSImage(example_tiff)

    # Analyze the image (placeholder for user)
    # result = image.analyze(roi=(50, 100, 50, 100))
    # print(f"Analysis result: {result}")

    # Load an existing calibration (if available)
    calibration_file = os.path.join(
        resources_dir, 'ebt_calibration_lot03172103_RED.pkl')
    calibration = Calibration.load(calibration_file)

    print(calibration.lot)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
