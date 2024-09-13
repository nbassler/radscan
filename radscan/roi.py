import roifile
import logging

# Set up logger
logger = logging.getLogger(__name__)


class ROI:
    """
    A class to handle loading ROIs from a file and converting them into a list of
    (left, right, top, bottom) tuples. Compatible with ROIs saved from ImageJ.

    Attributes:
        fn (str): The path to the ROI file.
        rois (list): A list of tuples representing each ROI as (left, right, top, bottom).

    Methods:
        load_rois(fn):
            Static method to load ROIs from an ImageJ-compatible file.
    """

    def __init__(self, fn):
        """
        Initializes the ROI class by loading ROIs from the specified file.

        Args:
            fn (str): The path to the ROI file.
        """
        self.fn = fn
        self.rois = self.load_rois(fn)

    @staticmethod
    def load_rois(fn):
        """
        Loads ROIs from an ImageJ-compatible file using roifile and converts them
        into a list of (left, right, top, bottom) tuples.

        The format is compatible with ROIs saved from ImageJ, including both single
        ROI files (.roi) and collections of ROIs stored in .zip format. ROIs are
        converted into a list of tuples where each tuple represents the bounding
        box of the ROI as (left, right, top, bottom).

        Args:
            fn (str): The path to the ROI file (can be a .roi or .zip file).

        Returns:
            list: A list of (left, right, top, bottom) tuples representing each ROI.
        """
        logger.debug(f"Loading ROIs from file: {fn}")

        roi_objs = roifile.roiread(fn)

        # Ensure the result is iterable, even if a single ROI is loaded.
        if not isinstance(roi_objs, list):
            roi_objs = [roi_objs]

        rois = [(r.left, r.right, r.top, r.bottom) for r in roi_objs]

        # Log the loaded ROIs for debugging purposes
        logger.debug(f"Loaded {len(rois)} ROIs from {fn}")
        for i, roi in enumerate(rois):
            logger.debug(f"ROI {i+1}: Left={roi[0]}, Right={roi[1]}, Top={roi[2]}, Bottom={roi[3]}")

        return rois
