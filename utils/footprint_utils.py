import cv2
import numpy as np


def threshold_footprint(
    footprint, percentile=99.9, threshold_type=cv2.THRESH_TOZERO
):
    """Adaptively threshold a cellular footprint image.

    :param footprint: footprint image matrix
    :param percentile: pixel value percentile to use for thresholding
    :param threshold_type: type of threshold to use

    :return thresh_img: threshold footprint image
    """
    # apply fixed threshold
    ret, thresh_img = cv2.threshold(footprint, 0, 100000, threshold_type)

    # compute weighted threshold (90% specified threshold, 10% image maximum)
    weighted_threshold = 0.9 * np.percentile(
        thresh_img, percentile
    ) + 0.1 * np.max(thresh_img)

    # threshold again using weighted threshold
    ret, thresh_img = cv2.threshold(
        thresh_img, weighted_threshold, 1, threshold_type
    )
    return thresh_img


def get_cell_contour(footprint):
    """Identify cell contour from footprint image.

    :param footprint: footprint image matrix
    :return cell_contour: contour of the cell
    """
    # threshold footprint image
    thresholded_footprint = threshold_footprint(footprint)

    # compute min-max of image
    min_val, max_val, min_index, max_indx = cv2.minMaxLoc(
        thresholded_footprint
    )
    if min_val == max_val:
        # no contrast so no contour can be found
        return []

    # pad image with one row of zeros all around to help with
    # delineation of cells against the border
    padded_img = cv2.copyMakeBorder(
        thresholded_footprint, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0
    )

    # threshold image to 0-1 range
    ret, thresh_img = cv2.threshold(padded_img, min_val, 1, cv2.THRESH_BINARY)

    # find cell contour
    contours, _ = cv2.findContours(
        thresh_img.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        # no contours found in processed footprint image
        return []

    # use largest contour by area as the cell contour
    cell_contour = contours[np.argmax([cv2.contourArea(c) for c in contours])]

    # cv2.findContours returns (y,x) coordinates
    # here we flip the order to (x,y) so that it matches the dimensions
    # of the input footprint image
    cell_contour = np.flip(cell_contour, axis=2)

    # offset contour points to account for padding needed to appropriately find contours
    cell_contour -= 1
    return cell_contour


def compute_cloud_center(points):
    """Compute the coordinates of the center of a cloud of points.

    :param points: list of points (x,y)
    :return centroid: center of the cloud of points
    """
    length = points.shape[0]
    if length == 0:
        return None
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    return sum_x / length, sum_y / length


def compute_contour_centroid(contour):
    """Compute the centroid of a given contour.

    :param contour: contour represented as a list of points
    :return : centroid of a contour
    """
    if len(contour) == 0:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        points = contour[:, 0, :]
        if len(points) == 0:
            return None
        else:
            return compute_cloud_center(points)
    centroid_x = moments["m10"] / moments["m00"]
    centroid_y = moments["m01"] / moments["m00"]
    return centroid_x, centroid_y


def compute_cell_centroid(footprint):
    """Compute the coordinates of the centroid of a given cell.

    :param footprint: footprint image matrix

    :return : coordinates of the centroid of the cell
    """
    # find cell contour
    cell_contour = get_cell_contour(footprint)

    # if no contour found, no centroid can be calculated
    if len(cell_contour) == 0:
        return None

    # compute centroid of the cell contour
    cell_centroid = compute_contour_centroid(cell_contour)

    # OpenCV's y-axis is (top: 0, bottom: footprint_height).
    # Here we reverse this so that centroid y coordinate is relative to the y=0 axis
    centroid_y = max(footprint.shape[1] - cell_centroid[1], 0)
    cell_centroid = (cell_centroid[0], centroid_y)
    return cell_centroid
