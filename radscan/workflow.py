import os
import logging

from radscan import RSImage, Calibration, NetOD


logger = logging.getLogger(__name__)


def analyze_simple_roi(pre_image, post_image, calibration_file=None):
    """
    Analyze the pre- and post-irradiation images by ROIs and return NetOD or dose (without background or control).

    Args:
        pre_image (RSImage): Pre-irradiation image with ROIs attached.
        post_image (RSImage): Post-irradiation image with corresponding ROIs attached.
        calibration_file (str, optional): Full path to the calibration file.

    Returns:
        list: A list of NetOD (or dose) values, one for each ROI.

    Raises:
        ValueError: If the number of ROIs in pre_image and post_image do not match.

    Notes:
        The number of ROIs in the pre- and post-irradiation images must match exactly. While the positions of the ROIs
        do not need to be identical, each corresponding ROI in the pre- and post-images must refer to the same strip
        of film. For example, pre_image.roi[0] must correspond to post_image.roi[0], pre_image.roi[1] must correspond
        to post_image.roi[1], and so on. The size and position of the ROIs may vary due to scanner placement, but
        the film strips are expected to match.
    """
    if len(pre_image.rois) != len(post_image.rois):
        logger.error(
            "Number of ROIs in pre-image does not match the number of ROIs in post-image.")
        logger.error(
            "Ensure that the ROIs in the pre- and post-images correspond to the same strips of film.")
        raise ValueError(
            "Mismatched ROI counts in pre- and post-irradiation images.")

    pre_values = pre_image.analyze()
    post_values = post_image.analyze()

    netod_values = []
    for pre_value, post_value in zip(pre_values, post_values):
        netod, _ = NetOD.simple(
            pre_value[0], post_value[0], pre_value[1], post_value[1])
        netod_values.append(netod)

    calibration = Calibration.load(
        calibration_file) if calibration_file else None
    return [calibration.dose(netod) for netod in netod_values] if calibration else netod_values


def analyze_simple_image(pre_image, post_image, calibration_file=None):
    """
    Analyze the entire post-irradiation image and return NetOD or dose (without background or control).

    Args:
        pre_image (RSImage): Pre-irradiation image with one or more ROIs attached.
        post_image (RSImage): Post-irradiation image (without ROIs).
        calibration_file (str, optional): Full path to the calibration file.

    Returns:
        np.ndarray: A 2D array of NetOD (or dose) values.
    """
    # Analyze pre-irradiation image and average their results over all attached ROIs
    pre_value = pre_image.analyze(single=True)

    # Calculate 2D NetOD using the scalar from pre-image and 2D pixel values from post-image
    netod, _ = NetOD.simple(pre_value[0], post_image.image)

    # Load calibration file if provided and convert NetOD to dose
    calibration = Calibration.load(
        calibration_file) if calibration_file else None

    return calibration.dose(netod) if calibration else netod


def analyze_roi(pre_image, post_image, control_pre_image, control_post_image, background_image, calibration_file=None):
    """
    Analyze the pre and post-irradiation images along with control and background,
    convert pixel values to NetOD or dose, and return a list of results for each ROI.

    Args:
        pre_image (RSImage): Pre-irradiation image with ROIs attached.
        post_image (RSImage): Post-irradiation image with ROIs attached.
        control_pre_image (RSImage): Control pre-irradiation image with ROIs attached.
        control_post_image (RSImage): Control post-irradiation image with ROIs attached.
        background_image (RSImage): Background image with ROIs attached.
        calibration_file (str, optional): Full path to the calibration file.

    Returns:
        list: A list of NetOD, or dose, values, one for each ROI.
    """

    # Check if the number of ROIs matches between pre and post images
    if len(pre_image.rois) != len(post_image.rois):
        logger.error(
            "Number of ROIs must match between pre and post-irradiation images.")
        logger.error(
            "Ensure that pre-image ROIs correspond to the same film strips in post-image.")
        return

    # Check if the number of ROIs matches between control pre and post images
    if len(control_pre_image.rois) != len(control_post_image.rois):
        logger.error(
            "Number of ROIs must match between control pre and post-irradiation images.")
        return

    # Analyze pre and post images for each ROI
    pre_values = pre_image.analyze()
    post_values = post_image.analyze()

    # Analyze control and background images as scalars (single=True)
    control_pre_value = control_pre_image.analyze(single=True)
    control_post_value = control_post_image.analyze(single=True)
    background_value = background_image.analyze(single=True)

    netod_values = []

    # Loop through each ROI in the pre and post images
    for i, (pre_value, post_value) in enumerate(zip(pre_values, post_values)):
        # Use NetOD.calc for full analysis with background and control
        netod, _ = NetOD.calc(pre_value[0], post_value[0],
                              control_pre_value[0], control_post_value[0],
                              background_value[0] if background_value else 0,
                              pre_value[1], post_value[1], control_pre_value[1], control_post_value[1],
                              background_value[1] if background_value else 0)
        netod_values.append(netod)

    # If calibration is provided, convert NetOD to dose
    calibration = Calibration.load(
        calibration_file) if calibration_file else None
    if calibration:
        dose_values = [calibration.dose(netod) for netod in netod_values]
        return dose_values
    else:
        return netod_values


def analyze_image(pre_image, post_image, control_pre_image, control_post_image, background_image, calibration_file=None):
    """
    Analyze the entire post-irradiation image along with control and background,
    convert pixel values to NetOD or dose, and return a 2D array of results.

    Args:
        pre_image (RSImage): Pre-irradiation image with ROIs attached.
        post_image (RSImage): Post-irradiation image (full image, without ROIs).
        control_pre_image (RSImage): Control pre-irradiation image with ROIs attached.
        control_post_image (RSImage): Control post-irradiation image with ROIs attached.
        background_image (RSImage): Background image with ROIs attached.
        calibration_file (str, optional): Full path to the calibration file.

    Returns:
        np.ndarray: A 2D array of NetOD (or dose) values.
    """

    # Analyze pre-image and control images, using the average of ROIs as a single scalar
    pre_value, spvb = pre_image.analyze(single=True)[0:2]
    control_pre_value, spvcb = control_pre_image.analyze(single=True)[0:2]
    control_post_value, spvca = control_post_image.analyze(single=True)[0:2]

    # Analyze background image as a scalar (if provided)
    if background_image:
        background_value, spvbk = background_image.analyze(single=True)[0:2]
    else:
        background_value = 0
        spvbk = 0

    # Analyze the full post-irradiation image (2D array)
    # Placeholder for post_image stderr
    post_image_values, spva = post_image.image, 0

    # Calculate 2D NetOD using the scalars from pre, control, and background images and the 2D post-irradiation values
    netod, _ = NetOD.calc(pre_value, post_image_values,
                          control_pre_value, control_post_value,
                          background_value, spvb, spva, spvcb, spvca, spvbk)

    # Load calibration file if provided and convert NetOD to dose
    calibration = Calibration.load(
        calibration_file) if calibration_file else None

    return calibration.dose(netod) if calibration else netod
