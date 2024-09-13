import numpy as np
import tifffile
import logging

logger = logging.getLogger(__name__)


class EBTImage:
    """
    A class to handle EBT film scans, load TIFF images, extract metadata, and work with ROIs.

    Attributes:
        fn (str): The filename of the TIFF image.
        rois (list): A list to store Region of Interest (ROI) specifications.
        image (np.ndarray): A NumPy array containing the image data, with shape
                            depending on the resolution and bit depth of the TIFF file.
                            For color images, the shape is typically (height, width, channels),
                            where channels correspond to Red, Green, and Blue (RGB).
        metadata (dict): A dictionary containing TIFF metadata extracted from the file.
                         This includes information like resolution, bit depth, compression, etc.

    Methods:
        analyze(rois=None, channel=0):
            Analyzes the image within specified ROIs or self.rois and computes statistics for each ROI.

        show(channel=0):
            Displays the entire image using matplotlib.

        show_rois(rois=None, channel=0):
            Displays the image for each specified ROI or self.rois using matplotlib.
    """

    def __init__(self, fn, rois=None):
        """
        Initializes the EBTImage class by loading the TIFF image, extracting metadata, and optionally setting ROIs.

        Args:
            fn (str): The path to the TIFF image file.
            rois (list, optional): A list of tuples specifying Regions of Interest (ROIs) as (xmin, xmax, ymin, ymax).
                                   Defaults to an empty list if no ROI is provided.
        """
        self.fn = fn
        self.rois = rois if rois is not None else []

        # Load image and metadata during initialization
        with tifffile.TiffFile(fn) as tif:
            self.image = tif.asarray()
            self.metadata = {tag.name: tag.value for tag in tif.pages[0].tags}

            # Log metadata for debugging purposes
            logger.debug(f"Loaded TIFF file: {self.fn}")
            logger.debug("Metadata:")
            for key, value in self.metadata.items():
                logger.debug(f"{key}: {value}")

    def analyze(self, rois=None, channel=0):
        """
        Analyze all or selected Regions of Interest (ROIs) within the image.

        Args:
            rois (list, optional): A list of tuples specifying ROIs as (xmin, xmax, ymin, ymax). If None,
                                   the method will use self.rois.
            channel (int): The color channel to analyze. Default is 0 (Red for color images).

        Returns:
            list: A list of tuples containing (mean, stderr, minval, maxval) for each ROI.
        """
        rois_to_analyze = rois if rois is not None else self.rois

        if not rois_to_analyze:
            raise ValueError("No ROIs provided or available in self.rois.")

        results = []
        for roi in rois_to_analyze:
            xmin, xmax, ymin, ymax = roi
            if xmin < 0 or xmax > self.image.shape[1] or ymin < 0 or ymax > self.image.shape[0]:
                raise ValueError(f"ROI {roi} is out of image bounds.")

            _image = self.image[ymin:ymax, xmin:xmax, :]
            _imc = _image[:, :, channel]
            mean = np.mean(_imc)
            stddev = np.std(_imc, ddof=1)
            stderr = stddev / np.sqrt(np.size(_imc))

            results.append((mean, stderr, np.min(_imc), np.max(_imc)))

        return results

    def show(self, channel=0):
        """
        Displays the entire image using matplotlib.

        Args:
            channel (int, optional): The color channel to display. Default is 0 (Red for color images).
        """
        import matplotlib.pyplot as plt

        _image = self.image[:, :, channel]
        plt.imshow(_image)
        plt.title(f"Entire Image: {self.fn}")
        plt.colorbar(label="Pixel Values")
        plt.show()

    def show_rois(self, rois=None, channel=0):
        """
        Displays the image for each specified ROI or self.rois using matplotlib.

        Args:
            rois (list, optional): A list of tuples specifying ROIs as (xmin, xmax, ymin, ymax). If None,
                                   the method will use self.rois.
            channel (int, optional): The color channel to display. Default is 0 (Red for color images).
        """
        import matplotlib.pyplot as plt

        rois_to_show = rois if rois is not None else self.rois

        if not rois_to_show:
            raise ValueError("No ROIs provided or available in self.rois.")

        for roi in rois_to_show:
            xmin, xmax, ymin, ymax = roi
            if xmin < 0 or xmax > self.image.shape[1] or ymin < 0 or ymax > self.image.shape[0]:
                raise ValueError(f"ROI {roi} is out of image bounds.")

            _image = self.image[ymin:ymax, xmin:xmax, :]
            _imc = _image[:, :, channel]

            plt.imshow(_imc)
            plt.title(f"ROI: {roi}")
            plt.colorbar(label="Pixel Values")
            plt.show()
