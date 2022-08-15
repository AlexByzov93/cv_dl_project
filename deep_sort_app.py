# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import *
from torchreid.utils.feature_extractor import *

import tensorflow as tf
import tensorflow_hub as hub

def extract_patches(image, boxes_scores):
    """
    extract patches from an image with box scores
    """
    patches = np.asarray([image[y1:y2, x1:x2] for y1, x1, y2, x2, _ in np.int32(boxes_scores)])
    return patches

def create_object_detector(model_det):
    """
    downloads object detector and loads it into a program, creates a specific transformation for images for object detection
    """
    object_detector_model = f"https://tfhub.dev/tensorflow/efficientdet/{model_det}/detection/1"
    object_detector = hub.load(object_detector_model)

    def detection_img_transformer(image):
        return tf.image.convert_image_dtype(image, tf.uint8)[tf.newaxis, ...]

    return object_detector, detection_img_transformer

def create_reid_extractor(model_reid):
    """
    creates feature extractor from image with different models trained for REID task
    """
    reid_feature_extractor = FeatureExtractor(model_reid)
    
    return reid_feature_extractor

def create_custom_detections(image, frame_idx, object_detector, detection_img_transformer, reid_feature_extractor):
    """
    creates a detection_mat np array in a similar to original format where first ten columns are from MOT challenge format and other are for features
    """
    det_image = detection_img_transformer(image)
    boxes, scores, _, _ = object_detector(det_image)
    boxes = boxes.numpy()
    boxes = boxes.reshape(boxes.shape[1], boxes.shape[2])
    scores = scores.numpy()
    scores = scores.reshape(scores.shape[1], 1)

    boxes_scores = np.concatenate((boxes, scores), axis=1)

    boxes_scores = boxes_scores[boxes_scores[:, 4] >= 0.1]

    box_rows, _ = boxes_scores.shape

    patches = extract_patches(image, boxes_scores)

    boxes_scores[:, 2] = boxes_scores[:, 2] - boxes_scores[:, 0]
    boxes_scores[:, 3] = boxes_scores[:, 3] - boxes_scores[:, 1]

    new_boxes_scores = boxes_scores[:, [1, 0, 3, 2, 4]]

    features = np.array([reid_feature_extractor(img).cpu().numpy() for img in patches])
    features = features.reshape(box_rows, features.shape[2])

    detection_mat = np.concatenate(
        (
            np.repeat(frame_idx, box_rows).reshape(box_rows, 1),
            np.repeat(-1, box_rows).reshape(box_rows, 1),
            new_boxes_scores,
            np.repeat(np.array([-1, -1, -1]), box_rows).reshape(box_rows, 3),
            features
        ),
        axis=1
    )

    return detection_mat



def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info

def create_detections(image, detection_mat, frame_idx, min_height=0, custom_detection=False, object_detector=None, detection_img_transformer=None, reid_feature_extractor=None):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    if custom_detection:
        detection_mat = create_custom_detections(
            image=image,
            frame_idx=frame_idx,
            object_detector=object_detector,
            detection_img_transformer=detection_img_transformer,
            reid_feature_extractor=reid_feature_extractor
            )

        detection_list = []
        for row in detection_mat:
            bbox, confidence, feature = row[2:6], row[6], row[10:]
            if bbox[3] < min_height:
                continue
            detection_list.append(Detection(bbox, confidence, feature))
        
    else:
        frame_indices = detection_mat[:, 0].astype(np.int)
        mask = frame_indices == frame_idx

        detection_list = []
        for row in detection_mat[mask]:
            bbox, confidence, feature = row[2:6], row[6], row[10:]
            if bbox[3] < min_height:
                continue
            detection_list.append(Detection(bbox, confidence, feature))
    return detection_list

def deep_sort_run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, custom_detection=False, model_det="lite0", model_reid="resnet18"):
    """Run multi-target tracker on a particular sequence.
    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    if custom_detection:
        object_detector, detection_img_transformer = create_object_detector(model_det)
        reid_feature_extractor = create_reid_extractor(model_reid)
    else:
        object_detector, detection_img_transformer, reid_feature_extractor = None, None, None


    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)
        image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
        # Load image and generate detections.
        detections = create_detections(image,
            seq_info["detections"], frame_idx, min_detection_height, custom_detection, object_detector, detection_img_transformer, reid_feature_extractor)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = image
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")