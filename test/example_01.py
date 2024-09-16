import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from radscan import RSImage, ROI, Calibration
from radscan.workflow import analyze_simple_roi, analyze_simple_image, analyze_image, analyze_roi

logger = logging.getLogger(__name__)


def main(args=None):
    """
    Main entry point for RadScan tool.
    """

    # Setup logging to show debug info
    logging.basicConfig(level=logging.INFO)

    # Filenames and directories

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

    # in this case there are several regoings of interest in the image.
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
    # However, the number of ROIs must match the number of ROIs in the pre-irradiation image.
    roi_post_filename = "RoiSet_post1.zip"

    # Calibration file, which is a pickle file containing the calibration data.
    # It is also possible to use the Calibration class to make a new set, but here we assume this is already done.
    calibration_file = "./resources/ebt_calibration_lot03172103_RED.pkl"
    channel = 0

    # plot calibration curve:
    # calibration = Calibration.load(calibration_file)
    # calibration.plot(save="calibration_curve_RED.png")

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

    # Simple analysis by ROI, which means, each ROI an average dose is calculated:
    results_by_roi = analyze_simple_roi(
        pre_image, post_image, calibration_file, channel)
    for idx, dose in enumerate(results_by_roi):
        logger.info(f"ROI {idx+1}: {dose:.2f} Gy")

    # But alternatively, we can also do a full image analysis, which means, the full post_image is converted from pixel_values to dose:
    results_by_image_netod = analyze_simple_image(
        pre_image, post_image, channel=channel)
    results_by_image_dose = analyze_simple_image(
        pre_image, post_image, calibration_file, channel)

    print(pre_image.rois)

    # exit()

    # Plot the full-image dose map
    # plot_results(results_by_image_netod, dpi=300,
    #             pixel_size=0.1, rois=roi_post.rois, vmax=1.0)
    # plot_results(results_by_image_dose, dpi=300,
    #              pixel_size=0.1, rois=roi_post.rois)
    # Plot the full-image dose map and save it to file
    # plot_results(results_by_image, dpi=300, pixel_size=0.1,
    #             save="dose_distribution.png")

    # now let us try the complicated method, here you also need background and control images.
    # background_image = RSImage([os.path.join(data_dir, "background.tif")])
    # control_pre_image = RSImage([os.path.join(data_dir, "control_pre.tif")])
    # control_post_image = RSImage([os.path.join(data_dir, "control_post.tif")])

    back_filenames = ["img20230504_09342060.tif",
                      "img20230504_09344731.tif",
                      "img20230504_09351256.tif",
                      "img20230504_09353762.tif"]
    background_image = RSImage([os.path.join(data_dir, fn)
                               for fn in back_filenames])

    # for the control images we will use the pre image for both pre and post control images.
    control_pre_image = pre_image
    control_post_image = pre_image

    roi_background_filename = "RoiSet_back.zip"
    roi_back = ROI(os.path.join(data_dir, roi_background_filename))
    background_image.rois = roi_back.rois

    # pre_image, post_image, control_pre_image, control_post_image, background_image, calibration_file=None, channel=0):
    results_by_image_netod = analyze_image(pre_image, post_image, control_pre_image,
                                           control_post_image, background_image, calibration_file, channel)
    # Plot the full-image dose map
    plot_results(results_by_image_netod, dpi=300,
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

    # Convert pixel indices to mm using the pixel size
    height_in_mm = results.shape[0] * pixel_size
    width_in_mm = results.shape[1] * pixel_size

    if not vmax:
        vmax = results.max()
        print(vmax)
        print(results.min())
        print(results.shape)

    if plot_type == "image":
        # Set vmin and vmax based on actual min/max values of the results array

        # Plot the full 2D dose map with proper vmin and vmax
        plt.imshow(results, vmin=0, vmax=vmax, cmap="gist_ncar")
        cb = plt.colorbar()
        cb.set_label("Dose [Gy]")

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
