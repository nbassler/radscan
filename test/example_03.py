import os
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from radscan import RSImage, ROI
from radscan import CHANNEL_MAP
from radscan import Calibration
from radscan.workflow import analyze_simple_image, analyze_simple_roi


logger = logging.getLogger(__name__)


def main(args=None):
    """
    Example of using the RadScan tool to build a calibration curve for the
    "simple analysis" workflow.
    """

    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    # Setup filenames and directories

    # working directory with the input images and ROIs are located:
    # test dataset 20230427_EBT.tar.gz will be provided somewhere in the future.
    data_dir = "/home/bassler/Desktop/20230427_EBT/"

    # first 4 scans of pre-irradiation images.
    pre_filenames_set1 = [
        "img20230424_13241513.tif",
        "img20230424_13243703.tif",
        "img20230424_13245545.tif",
        "img20230424_13251336.tif"
    ]
    # in this case there are several regions of interest in the image.
    # the ROIs we marked in ImageJ, and then saved by ImageJ as a zip file.
    # should it be a single ROI, then it has suffix .roi, which is also OK.
    roi_pre_filename_set1 = "RoiSet_pre1.zip"

    # post irradiation images, again 4 scans of the same area
    post_filenames_set1 = [
        "img20230427_11184159.tif",
        "img20230427_11190007.tif",
        "img20230427_11191783.tif",
        "img20230427_11193478.tif"
    ]
    # ROI size and positions do not have to match the pre-irradiation ROIs.
    # However, the _number_ (that is the index) of each ROIs
    # must match the number (index) of ROIs in the pre-irradiation image.
    # The length of the ROI list must be the same in both pre and post images.
    roi_post_filename_set1 = "RoiSet_post1.zip"

    # Same again, for images of second set of doses:
    pre_filenames_set2 = [
        "img20230424_13273415.tif",
        "img20230424_13281045.tif",
        "img20230424_13283343.tif",
        "img20230424_13285512.tif"
    ]
    roi_pre_filename_set2 = "RoiSet_pre2.zip"
    post_filenames_set2 = [
        "img20230427_11261787.tif",
        "img20230427_11264268.tif",
        "img20230427_11271797.tif",
        "img20230427_11281967.tif"
    ]
    roi_post_filename_set2 = "RoiSet_post2.zip"

    # list of dose values for each ROI:
    doses_set1 = [12, 20, 2, 8, 2, 20, 4, 12, 4, 0]
    doses_set2 = [10, 3, 16, 6, 5, 10, 16, 8, 5, 3, 6]

    # select channel for making the calibration curve
    channel = 2  # 0=RED channel, 1=GREEN, 2=BLUE

    # Next we need to calculate NetOD values from the images.

    # Load images
    pre_image_set1 = RSImage([os.path.join(data_dir, fn)
                             for fn in pre_filenames_set1])
    post_image_set1 = RSImage([os.path.join(data_dir, fn)
                              for fn in post_filenames_set1])
    pre_image_set2 = RSImage([os.path.join(data_dir, fn)
                             for fn in pre_filenames_set2])
    post_image_set2 = RSImage([os.path.join(data_dir, fn)
                              for fn in post_filenames_set2])

    # Load ROIs
    roi_pre_set1 = ROI(os.path.join(data_dir, roi_pre_filename_set1))
    roi_post_set1 = ROI(os.path.join(data_dir, roi_post_filename_set1))
    roi_pre_set2 = ROI(os.path.join(data_dir, roi_pre_filename_set2))
    roi_post_set2 = ROI(os.path.join(data_dir, roi_post_filename_set2))

    # and attach the ROI tuple lists to the images:
    pre_image_set1.rois = roi_pre_set1.rois
    post_image_set1.rois = roi_post_set1.rois
    pre_image_set2.rois = roi_pre_set2.rois
    post_image_set2.rois = roi_post_set2.rois

    # Now we have all input data available, so we can proceed with the analysis.
    # First we do a simple analysis by ROI, which means, each ROI an average netOD is calculated.
    # Note, since we do not provide a calibration file, the results will be in netOD units.
    results_by_roi_set1 = analyze_simple_roi(
        pre_image_set1, post_image_set1, channel=channel)

    results_by_roi_set2 = analyze_simple_roi(
        pre_image_set2, post_image_set2, channel=channel)

    # For checking netODs, we can also plot 2D images, just to check them:
    results_by_image_netod_set1 = analyze_simple_image(
        pre_image_set1, post_image_set1, channel=channel)
    results_by_image_netod_set2 = analyze_simple_image(
        pre_image_set2, post_image_set2, channel=channel)

    # Plot the full-image dose map
    plot_results(results_by_image_netod_set1, dpi=300,
                 pixel_size=0.1, rois=roi_post_set1.rois, vmax=1.0)  # Caveat: units are NetOD
    plot_results(results_by_image_netod_set2, dpi=300,
                 pixel_size=0.1, rois=roi_post_set2.rois, vmax=1.0)  # Caveat: units are NetOD

    doses = doses_set1 + doses_set2
    netODs = results_by_roi_set1 + results_by_roi_set2
    # Create a Calibration object and save it
    calib = Calibration(ds=doses, nods=netODs, lot="03172103",
                        channel=CHANNEL_MAP[channel])
    calib_fn = f"resources/ebt_calibration_lot03172103_{CHANNEL_MAP[channel]}_simple.pkl"
    calib.save(filename=calib_fn)
    logger.info(f"Calibration saved as {calib_fn}")

    # Plot the calibration curve for visual inspection
    calib.plot()


# first 4 scans of pre-irradiation images.

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
    height_in_mm = results.shape[0] * pixel_size
    width_in_mm = results.shape[1] * pixel_size

    if not vmax:
        vmax = results.max()  # TODO: eliminate NaN values from results array

    if plot_type == "image":
        # Plot the full 2D dose map with proper vmin and vmax
        plt.imshow(results, vmin=0, vmax=vmax, cmap="gist_ncar")
        cb = plt.colorbar()
        cb.set_label("NetOD []")

        plt.xlabel(f"X axis [pixels]")
        plt.ylabel(f"Y axis [pixels]")
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
