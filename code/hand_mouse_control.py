from utils import detector_utils as detector_utils
from utils import visualization_utils as vis_utils
from utils import label_map_util
import cv2
import os
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool, Value, Process
import time
from utils.detector_utils import WebcamVideoStream, mouse_move_worker
import datetime
import argparse

score_thresh = 0.27
num_hands = 2

def worker(input_q, output_q, frame_processed, objectX, objectY):
    print(">> chargement du frozen model dans le worker")

    MODEL_NAME = 'hand_inference_graph'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

    NUM_CLASSES = 1

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    while True:
        frame = input_q.get()
        if (frame is not None):
            # detection
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # mouse deplacement
            if len(boxes) > 0 and scores[0] > 0.27:
                b = boxes[0]
                objectX.value, objectY.value = (b[1]+b[3])/2, (b[0]+b[2])/2
            else:
                objectX.value, objectY.value = 0.0, 0.0
            # dessin box
            vis_utils.visualize_boxes_and_labels_on_image_array(
                frame, boxes, classes, scores, category_index, max_boxes_to_draw=num_hands, min_score_thresh=score_thresh, use_normalized_coordinates=True, line_thickness=4)
            output_q.put(frame)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()


if __name__ == '__main__':

    input_q = Queue(maxsize=1)
    output_q = Queue(maxsize=1)

    video_capture = WebcamVideoStream(src=0,
                                      width=300,
                                      height=200).start()

    frame_processed = 0
    objectX, objectY = Value('d', 0.0), Value('d', 0.0)
    mouse_process = Process(target=mouse_move_worker, args=(objectX, objectY))
    mouse_process.start()

    pool = Pool(1, worker,
                (input_q, output_q, frame_processed, objectX, objectY))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    while True:
        frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        index += 1

        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_frame = output_q.get()

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        num_frames += 1

        if (output_frame is not None):
            cv2.imshow('Hand Mouse Control', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    elapsed_time = (datetime.datetime.now() -
                    start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("fps", fps)
    pool.terminate()
    mouse_process.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
