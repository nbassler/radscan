import os
import copy
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from radscan import RSImage, ROI
# from radscan import Calibration # only needed if we want to plot the calibration curve
from radscan.workflow import analyze_image, analyze_roi

logger = logging.getLogger(__name__)


def main(args=None):
    """
    Example of using the RadScan tool to analyze EBT film images.
    This is the "complex" example, which includes background correction,
    control films, and full-image analysis.
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

    # background images, which are used to correct the images for background noise.
    # These are scans of a black film, which is used to calibrate the scanner.
    back_filenames = [
        "img20230504_09342060.tif",
        "img20230504_09344731.tif",
        "img20230504_09351256.tif",
        "img20230504_09353762.tif"
    ]

    # ROI is simply a square area of the black film, which should match the area where the films are scanned.
    # The background images are taken without any film in the scanner.
    roi_background_filename = "RoiSet_back.zip"

    # Control films before and after irradiation are included in the pre_ and post_filenames.
    # So we are not loading any more films by now.

    # Next, set the calibration file, which is a pickle file containing a calibration dataset
    # It is also possible to use the Calibration class to make a new set, but for this example it is already done.
    calibration_file = "./resources/ebt_calibration_lot03172103_RED.pkl"
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

    # We need also a control film, which is a film that is not irradiated.
    # This is used to correct for scanner variations.
    # The control images are taken before and after the irradiation.

    # in this case, the pre and post data is stored in the same file, but in a single ROI.
    control_pre_image = copy.deepcopy(pre_image)
    control_post_image = copy.deepcopy(post_image)
    # the last ROI is the control ROI:
    control_pre_image.rois = [pre_image.rois[-1]]
    control_post_image.rois = [post_image.rois[-1]]

    # If the control film was not made, then pre-scanned films can be used for both parts.
    # In this case, the pre_image is used for both control images.
    # This is not ideal, but can be used as a workaround:
    # control_pre_image = pre_image
    # control_post_image = pre_image

    # The background images are used to correct the images for background noise.
    background_image = RSImage([os.path.join(data_dir, fn)
                                for fn in back_filenames])

    roi_back = ROI(os.path.join(data_dir, roi_background_filename))
    background_image.rois = roi_back.rois

    # Now we have all input data available, so we can proceed with the analysis.
    # First we do a simple analysis by ROI, which means, each ROI an average dose is calculated:
    results_by_roi = analyze_roi(pre_image, post_image, control_pre_image,
                                 control_post_image, background_image,
                                 calibration_file, channel)
    # check against nominal doses:
    dose_nominal = [12, 20, 2, 8, 2, 20, 4, 12, 4, 0]
    for idx, dose in enumerate(results_by_roi):
        logger.info(f"ROI {idx+1:02}: {dose_nominal[idx]:8.2f} {dose:8.2f} Gy")

    # But alternatively, we can also do a full image analysis,
    # which means, the full post_image is converted from pixel_values to dose, using the calibration curve:
    results_by_image_dose = analyze_image(pre_image, post_image,
                                          control_pre_image,
                                          control_post_image, background_image,
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
