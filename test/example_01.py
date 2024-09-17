import os
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from radscan import RSImage, ROI
# from radscan import Calibration # only needed if we want to plot the calibration curve
from radscan.workflow import analyze_simple_image, analyze_simple_roi

logger = logging.getLogger(__name__)


def main(args=None):
    """
    Example of using the RadScan tool to analyze EBT film images.
    This is the simple example, which does not include background correction
    and control films.
    """

    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    # Setup filenames and directories

    # working directory with the input images and ROIs are located:
    # test dataset 20230427_EBT.tar.gz will be provided somewhere in the future.
    data_dir = "/home/bassler/Desktop/20230427_EBT/"

    # first 4 scans of pre-irradiation images.
    pre_filenames = [
        "img20230424_13241513.tif",
        "img20230424_13243703.tif",
        "img20230424_13245545.tif",
        "img20230424_13251336.tif"
    ]
    # in this case there are several regions of interest in the image.
    # the ROIs we marked in ImageJ, and then saved by ImageJ as a zip file.
    # should it be a single ROI, then it has suffix .roi, which is also OK.
    roi_pre_filename = "RoiSet_pre1.zip"

    # post irradiation images, again 4 scans of the same area
    post_filenames = [
        "img20230427_11184159.tif",
        "img20230427_11190007.tif",
        "img20230427_11191783.tif",
        "img20230427_11193478.tif"
    ]
    # ROI size and positions do not have to match the pre-irradiation ROIs.
    # However, the _number_ (that is the index) of each ROIs
    # must match the number (index) of ROIs in the pre-irradiation image.
    # The length of the ROI list must be the same in both pre and post images.
    roi_post_filename = "RoiSet_post1.zip"

    # Next, set the calibration file, which is a pickle file containing a calibration dataset
    # It is also possible to use the Calibration class to make a new set, but for this example it is already done.
    calibration_file = "./resources/ebt_calibration_lot03172103_RED_simple.pkl"
    channel = 0  # 0=RED channel, 1=GREEN, 2=BLUE

    # If you want to plot the calibration curve, you can do so with the following line:
    # calibration = Calibration.load(calibration_file)
    # calibration.plot()

    # Load images
    pre_image = RSImage([os.path.join(data_dir, fn) for fn in pre_filenames])
    post_image = RSImage([os.path.join(data_dir, fn) for fn in post_filenames])

    # In case no ROIs were made with ImageJ, it can also directly be can also be set manually
    # These are a list of tuples, for example for 3 ROIs:
    # roi_pre.rois = [(50, 100, 50, 100), (150, 200, 150, 200), (250, 300, 250, 300)]
    # Here, however, we load the ROIs from the ImageJ zip file:
    roi_pre = ROI(os.path.join(data_dir, roi_pre_filename))
    roi_post = ROI(os.path.join(data_dir, roi_post_filename))

    # and attach the ROI tuple lists to the images:
    pre_image.rois = roi_pre.rois
    post_image.rois = roi_post.rois

    # Now we have all input data available, so we can proceed with the analysis.
    # First we do a simple analysis by ROI, which means, each ROI an average dose is calculated:
    results_by_roi = analyze_simple_roi(
        pre_image, post_image, calibration_file, channel)

    # check against nominal doses:
    dose_nominal = [12, 20, 2, 8, 2, 20, 4, 12, 4, 0]
    for idx, dose in enumerate(results_by_roi):
        logger.info(f"ROI {idx+1:02}: {dose_nominal[idx]:8.2f} {dose:8.2f} Gy")

    # But alternatively, we can also do a full 2D-image analysis,
    # which means, the full post_image is converted from pixel_values to dose, using the calibration curve:
    results_by_image_dose = analyze_simple_image(pre_image, post_image,
                                                 calibration_file, channel)

    # Plot the full-image dose map
    plot_results(results_by_image_dose, dpi=300,
                 pixel_size=0.1, rois=roi_post.rois, vmax=22.0)


def plot_results(results, dpi, pixel_size, plot_type="image", save=None, rois=None, vmax=None):
    """
    Function to plot results of the analysis.

    Args:
        results (np.ndarray or list): The results to plot. Can be either a 2D array for full-image analysis
                                      or a list of values for ROI-based analysis.
        dpi (float): The dots per inch (DPI) resolution of the scan.
        pixel_size (float): The size of each pixel in millimeters.
        plot_type (str): Either "image" for 2D analysis or "roi" for ROI-based analysis.
        save (str, optional): If provided, the plot will be saved to the given filename instead of being displayed.
    """

    # TODO: Convert pixel indices to mm using the pixel size
    # height_in_mm = results.shape[0] * pixel_size
    # width_in_mm = results.shape[1] * pixel_size

    if not vmax:
        vmax = results.max()  # TODO: eliminate NaN values from results array

    if plot_type == "image":
        # Plot the full 2D dose map with proper vmin and vmax
        plt.imshow(results, vmin=0, vmax=vmax, cmap="gist_ncar")
        cb = plt.colorbar()
        cb.set_label("Dose [Gy]")

        plt.xlabel("X axis [pixels]")
        plt.ylabel("Y axis [pixels]")
        # TODO: set axis scales to mm instead of pixels
        plt.title("Dose Distribution")
        plt.gca().set_aspect('auto')
        # Plot ROI rectangles
        if rois:
            for idx, (xmin, xmax, ymin, ymax) in enumerate(rois):
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                # Add ROI index to the center of the rectangle
                plt.text((xmin + xmax) / 2, (ymin + ymax) / 2, f"ROI {idx+1}",
                         color='red', ha='center', va='center', fontsize=8, fontweight='bold')

    elif plot_type == "roi":
        # Plot ROI-based dose results as a bar chart
        plt.bar(range(len(results)), results)
        plt.xlabel("ROI Index")
        plt.ylabel("Dose [Gy]")
        plt.title("Dose per ROI")
    else:
        logger.error(f"Unknown plot_type: {plot_type}")
        return

    # Save plot to file if filename is provided, else display it
    if save:
        plt.savefig(save, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved as {save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
