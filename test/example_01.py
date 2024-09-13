import os
import logging
import matplotlib.pyplot as plt
from radscan import RSImage, ROI
from radscan.workflow import analyze_simple_roi, analyze_simple_image

logger = logging.getLogger(__name__)


def main(args=None):
    """
    Main entry point for RadScan tool.
    """

    # Setup logging to show debug info
    logging.basicConfig(level=logging.DEBUG)

    # Filenames and directories
    pre_filenames = [
        "img20230424_13241513.tif",
        "img20230424_13243703.tif",
        "img20230424_13245545.tif",
        "img20230424_13251336.tif"
    ]
    post_filenames = [
        "img20230427_11184159.tif",
        "img20230427_11190007.tif",
        "img20230427_11191783.tif",
        "img20230427_11193478.tif"
    ]

    roi_pre_filename = "RoiSet_pre1.zip"
    roi_post_filename = "RoiSet_post1.zip"

    calibration_file = "./resources/ebt_calibration_lot03172103_RED.pkl"
    data_dir = "/home/bassler/Desktop/20230427_EBT/"

    # Load images
    pre_image = RSImage([os.path.join(data_dir, fn) for fn in pre_filenames])
    post_image = RSImage([os.path.join(data_dir, fn) for fn in post_filenames])

    # Load ROIs and attach them to the images
    roi_pre = ROI(os.path.join(data_dir, roi_pre_filename))
    roi_post = ROI(os.path.join(data_dir, roi_post_filename))
    # can also be set manually, if no ROI file is available
    pre_image.rois = roi_pre.rois
    post_image.rois = roi_post.rois

    # Simple analysis by ROI
    results_by_roi = analyze_simple_roi(
        pre_image, post_image, calibration_file)
    print(f"Results by ROI: {results_by_roi}")

    # Simple analysis by 2D image
    results_by_image = analyze_simple_image(
        pre_image, post_image, calibration_file)

    # Plot the full-image results and save it to file
    plot_results(results_by_image, dpi=300, pixel_size=0.1,
                 save="dose_distribution.png")


def plot_results(results, dpi, pixel_size, plot_type="image", save=None):
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

    # Convert pixel indices to mm using the pixel size
    height_in_mm = results.shape[0] * pixel_size
    width_in_mm = results.shape[1] * pixel_size

    if plot_type == "image":
        # Plot the full 2D dose map
        plt.imshow(results, vmin=0.0, vmax=22.0, cmap="gist_ncar")
        cb = plt.colorbar()
        cb.set_label("Dose [Gy]")
        plt.xlabel(f"X axis [pixels]")
        plt.ylabel(f"Y axis [pixels]")
        # TODO: set axis scales to mm instead of pixels
        plt.title("Dose Distribution")
        plt.gca().set_aspect('auto')

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
