from scipy import ndimage
import numpy as np


def dice_score(prediction, groundtruth, smooth=1.):
    numer = (prediction * groundtruth).sum()
    denor = (prediction + groundtruth).sum()
    # loss = (2 * numer + self.smooth) / (denor + self.smooth)
    dice = (2 * numer + smooth) / (denor + smooth)
    return dice


def lesion_wise_tp_fp_fn(truth, prediction, overlap_ratio=0.1):
    """
    Computes the true positives, false positives, and false negatives two masks. Masks are considered true positives
    if there is at least `overlap_ratio` overlap between the truth and the prediction.
    i.e. if overlap_ratio = 0.1, then at least 10% of the lesion voxels should overlap between the truth and 
    the prediction to be considered as true positive.
    Adapted from: https://github.com/npnl/atlas2_grand_challenge/blob/main/isles/scoring.py#L341

    Parameters
    ----------
    truth : array-like, bool
        3D array. If not boolean, will be converted.
    prediction : array-like, bool
        3D array with a shape matching 'truth'. If not boolean, will be converted.
    empty_value : scalar, float
        Optional. Value to which to default if there are no labels. Default: 1.0.

    Returns
    -------
    tp (int): 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
    fp (int): 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn (int): 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.

    Notes
    -----
    This function computes lesion-wise score by defining true positive lesions (tp), false positive lesions (fp) and
    false negative lesions (fn) using 3D connected-component-analysis.

    tp: 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
    fp: 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn: 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.
    """
    tp, fp, fn = 0, 0, 0

    # For each true lesion, check if at least a threshold overlap_ratio of the lesion voxels overlap with the prediction.
    # This determines true positives and false negatives (unpredicted lesions)
    labeled_ground_truth, num_truth_lesions = ndimage.label(truth.astype(bool))
    for idx_lesion in range(1, num_truth_lesions + 1):
        lesion = labeled_ground_truth == idx_lesion
        num_truth_lesion_voxels = np.sum(lesion)  # Total number of voxels in the GT lesion
        overlapping_voxels = np.sum(lesion * prediction)  # Number of GT voxels that overlap with the prediction
        # Check if at least 10% of the lesion voxels overlap with the prediction
        if overlapping_voxels / num_truth_lesion_voxels >= overlap_ratio:
            tp += 1
        else:
            fn += 1

    # For each predicted lesion, check if there is at least one overlapping voxel in the ground truth.
    labeled_prediction, num_pred_lesions = ndimage.label(prediction.astype(bool))
    for idx_lesion in range(1, num_pred_lesions+1):
        lesion = labeled_prediction == idx_lesion
        # num_pred_lesion_voxels = np.sum(lesion)
        # overlapping_voxels = np.sum(lesion & truth)
        # if overlapping_voxels / num_pred_lesion_voxels < self.overlap_ratio:
        #     fp += 1
        lesion_pred_sum = lesion + truth
        if(np.max(lesion_pred_sum) <= 1):  # No overlap
            fp += 1

    return tp, fp, fn

def lesion_f1_score(truth, prediction, overlap_ratio=0.1):
    """
    Computes the lesion-wise F1-score between two masks by defining true positive lesions (tp), false positive lesions (fp)
    and false negative lesions (fn) using 3D connected-component-analysis.

    Masks are considered true positives if at least one voxel overlaps between the truth and the prediction.

    Returns
    -------
    f1_score : float
        Lesion-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value
    """
    empty_value = 1.0   # Value to which to default if there are no labels. Default: 1.0.

    if not np.any(truth) and not np.any(prediction):
        # Both reference and prediction are empty --> model learned correctly
        return 1.0
    elif np.any(truth) and not np.any(prediction):
        # Reference is not empty, prediction is empty --> model did not learn correctly (it's false negative)
        return 0.0
    # if the ref is empty and prediction is empty --> it's false positive
    elif not np.any(truth) and np.any(prediction):
        return 0.0
    # if both are not empty, it's true positive
    else:
        tp, fp, fn = lesion_wise_tp_fp_fn(truth, prediction, overlap_ratio)
        f1_score = empty_value

        # Compute f1_score
        denom = tp + (fp + fn)/2
        if(denom != 0):
            f1_score = tp / denom
        return f1_score

def lesion_ppv(truth, prediction, overlap_ratio=0.1):
    """
    Computes the lesion-wise positive predictive value (PPV) between two masks
    Returns
    -------
    ppv (float): Lesion-wise positive predictive value as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value
    """
    if not np.any(truth) and not np.any(prediction):
        # Both reference and prediction are empty --> model learned correctly
        return 1.0
    elif np.any(truth) and not np.any(prediction):
        # Reference is not empty, prediction is empty --> model did not learn correctly (it's false negative)
        return 0.0
    # if the predction is not empty and ref is empty, it's false positive
    elif not np.any(truth) and np.any(prediction):
        return 0.0
    # if both are not empty, it's true positive
    else:
        tp, fp, _ = lesion_wise_tp_fp_fn(truth, prediction, overlap_ratio)
        ppv = 1.0

        # Compute ppv
        denom = tp + fp
        # denom should ideally not be zero inside this else as it should be caught by the empty checks above
        if(denom != 0):
            ppv = tp / denom
        return ppv

def lesion_sensitivity(truth, prediction, overlap_ratio=0.1):
    """
    Computes the lesion-wise sensitivity between two masks
    Returns
    -------
    sensitivity (float): Lesion-wise sensitivity as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value
    """
    empty_value = 1.0   # Value to which to default if there are no labels. Default: 1.0.

    if not np.any(truth) and not np.any(prediction):
        # Both reference and prediction are empty --> model learned correctly
        return 1.0
    # if the predction is not empty and ref is empty, it's false positive
    # if both are not empty, it's true positive
    else:

        tp, _, fn = lesion_wise_tp_fp_fn(truth, prediction, overlap_ratio)
        sensitivity = empty_value

        # Compute sensitivity
        denom = tp + fn
        if(denom != 0):
            sensitivity = tp / denom
        return sensitivity


class MorphologyOps(object):
    """
    Class that performs the morphological operations needed to get notably
    connected component. To be used in the evaluation
    """

    def __init__(self, binary_img, connectivity):
        self.binary_map = np.asarray(binary_img, dtype=np.int8)
        self.connectivity = connectivity

    def border_map(self):
        """
        Create the border map defined as the difference between the original image 
        and its eroded version

        :return: border
        """
        eroded = ndimage.binary_erosion(self.binary_map)
        border = self.binary_map - eroded
        return border

    def border_map2(self):
        """
        Creates the border for a 3D image
        :return:
        """
        west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
        east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
        north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
        south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
        top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
        bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)
        cumulative = west + east + north + south + top + bottom
        border = ((cumulative < 6) * self.binary_map) == 1
        return border

    def foreground_component(self):
        return ndimage.label(self.binary_map)

    def list_foreground_component(self):
        labels, _ = self.foreground_component()
        list_ind_lab = []
        list_volumes = []
        list_com = []
        list_values = np.unique(labels)
        for f in list_values:
            if f > 0:
                tmp_lab = np.where(
                    labels == f, np.ones_like(labels), np.zeros_like(labels)
                )
                list_ind_lab.append(tmp_lab)
                list_volumes.append(np.sum(tmp_lab))
                list_com.append(ndimage.center_of_mass(tmp_lab))
        return list_ind_lab, list_volumes, list_com


def border_distance(truth, pred, pixdim, connectivity=1):
        """
        This functions determines the map of distance from the borders of the
        prediction and the reference and the border maps themselves

        :return: distance_border_ref, distance_border_pred, border_ref,
        border_pred
        """
        border_truth = MorphologyOps(truth, connectivity).border_map()
        border_pred = MorphologyOps(pred, connectivity).border_map()
        oppose_truth = 1 - truth
        oppose_pred = 1 - pred
        distance_truth = ndimage.distance_transform_edt(
            1 - border_truth, sampling=pixdim
        )
        distance_pred = ndimage.distance_transform_edt(
            1 - border_pred, sampling=pixdim
        )
        distance_border_pred = border_truth * distance_pred
        distance_border_truth = border_pred * distance_truth
        return distance_border_truth, distance_border_pred, border_truth, border_pred


def normalised_surface_distance(truth, prediction, pixdim, overlap_ratio=0.1, tau=1):
        """
        Calculates the normalised surface distance (NSD) between prediction and reference
        using the distance parameter :math:`{\\tau}`

        Stanislav Nikolov, Sam Blackwell, Alexei Zverovitch, Ruheena Mendes, Michelle Livne, Jeffrey De Fauw, Yojan Patel,
        Clemens Meyer, Harry Askham, Bernadino Romera-Paredes, et al. 2021. Clinically applicable segmentation of head
        and neck anatomy for radiotherapy: deep learning algorithm development and validation study. Journal of Medical
        Internet Research 23, 7 (2021), e26151.

        .. math::

            NSD(A,B)^{(\\tau)} = \dfrac{|S_{A} \cap Bord_{B,\\tau}| + |S_{B} \cup Bord_{A,\\tau}|}{|S_{A}| + S_{B}}

        :return: NSD
        """
        if not np.any(truth) and not np.any(prediction):
            # Both reference and prediction are empty --> model learned correctly --> distance is 0
            return 0.0
        
        dist_ref, dist_pred, border_ref, border_pred = border_distance(truth, prediction, pixdim)
        reg_ref = np.where(
            dist_ref <= tau, np.ones_like(dist_ref), np.zeros_like(dist_ref)
        )
        reg_pred = np.where(
            dist_pred <= tau, np.ones_like(dist_pred), np.zeros_like(dist_pred)
        )
        numerator = np.sum(border_pred * reg_ref) + np.sum(border_ref * reg_pred)
        denominator = np.sum(border_ref) + np.sum(border_pred)
        return numerator / denominator