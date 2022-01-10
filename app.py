"""
Run a rest API exposing the yolov5s object detection model
"""

import numpy as np
import os
import random
import math
import argparse
import io
import cv2
import time
import torch

from flask import Flask, request, Response
from PIL import Image
from itertools import combinations

# Using feed from local webcam


app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5s"


def is_close(p1, p2):
    """
    #================================================================
    # 1. Purpose : Calculate Euclidean Distance between two points
    #================================================================    
    :param:
    p1, p2 = two points for calculating Euclidean Distance

    :return:
    dst = Euclidean Distance between two 2d points
    """
    dst = math.sqrt(p1 ** 2 + p2 ** 2)
    # =================================================================#
    return dst


def gen_frames(cap):
    start_time = time.time()
    display_time = 2
    fps = 0
    while True:
        success, frame = cap.read()
        detections = model(frame, size=640)
        detections = detections.pandas().xyxy[0].to_dict(orient="records")

        if len(detections) > 0:  # At least 1 detection in the image and check detection presence in a frame
            centroid_dict = dict()  # Function creates a dictionary and calls it centroid_dict
            objectId = 0  # We inialize a variable called ObjectId and set it to 0
            for detection in detections:  # In this if statement, we filter all the detections for persons only
                # Check for the only person name tag 
                name_tag = detection['name']  # Coco file has string of all the names
                if name_tag == 'person':
                    xmin, ymin, xmax, ymax = detection['xmin'], detection['ymin'], detection['xmax'], detection[
                        'ymax']  # Store the center points of the detections
                    x = int(xmin + (xmax - xmin) / 2)
                    y = int(ymin + (ymax - ymin) / 2)
                    cv2.circle(frame, (x, y), int(((xmax - xmin) * 0.05)), (0, 0, 255), -1)
                    centroid_dict[objectId] = (x, y, int(xmin), int(ymin), int(xmax),
                                               int(ymax))  # Create dictionary of tuple with 'objectId' as the index center points and bbox
                    objectId += 1  # Increment the index for each detection

            # =================================================================#

            # =================================================================
            # 3.2 Purpose : Determine which person bbox are close to each other
            # =================================================================
            red_zone_list = []  # List containing which Object id is in under threshold distance condition.
            red_line_list = []
            for (id1, p1), (id2, p2) in combinations(centroid_dict.items(),2):  # Get all the combinations of close detections, #List of multiple items - id1 1, points 2, 1,3
                dx, dy = p1[0] - p2[0], p1[1] - p2[1]  # Check the difference between centroid x: 0, y :1
                distance = is_close(dx, dy)  # Calculates the Euclidean distance
                if distance < 300.0:  # Set our social distance threshold - If they meet this condition then..
                    if id1 not in red_zone_list:
                        red_zone_list.append(id1)  # Add Id to a list
                        red_line_list.append(p1[0:2])  # Add points to the list
                    if id2 not in red_zone_list:
                        red_zone_list.append(id2)  # Same for the second id
                        red_line_list.append(p2[0:2])
                print(p1, '####', p2, '####', distance)
            for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
                if idx in red_zone_list:  # if id is in red zone list
                    print(len(red_zone_list))
                    cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255),
                                  2)  # Create Red bounding boxes  #starting point, ending point size of 2
                else:
                    cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0),
                                  2)  # Create Green bounding boxe bounding boxes  #starting
            # =================================================================
            # 3.3 Purpose : Display Risk Analytics and Show Risk Indicators
            # =================================================================
            text = "People at Risk: %s" % str(len(red_zone_list))  # Count People at Risk
            location = (10, 25)  # Set the location of the displayed text
            cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246, 86, 86), 2,
                        cv2.LINE_AA)  # Display Text

            for check in range(0,
                               len(red_line_list) - 1):  # Draw line between nearby bboxes iterate through redlist items
                start_point = red_line_list[check]
                end_point = red_line_list[check + 1]
                check_line_x = abs(end_point[0] - start_point[0])  # Calculate the line coordinates for x
                check_line_y = abs(end_point[1] - start_point[1])  # Calculate the line coordinates for y
                if (check_line_x < 75) and (
                        check_line_y < 25):  # If both are We check that the lines are below our threshold distance.
                    cv2.line(frame, start_point, end_point, (255, 0, 0),
                             2)  # Only above the threshold lines are displayed.
        # =================================================================#

        # cv2.imshow("webcam", frame)
        fps += 1
        TIME = time.time() - start_time
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWidnows()
            break

        if TIME > display_time:
            resp = "FPS:" + str(fps / TIME)
            fps = 0
            start_time = time.time()

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')


@app.route("/", methods=['GET'])
def status():
    return 'yolov5 object detection active'


@app.route("/detect", methods=['GET'])
def detect():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("We cannot open webcam")
    else:
        return Response(gen_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')
    cap.release()
    cv2.destroyAllWindows()


@app.route(DETECTION_URL, methods=["POST", "GET"])
def predict():
    if not request.method == "POST":
        return 'This is not a POST request'

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()

        img = Image.open(io.BytesIO(image_bytes))

        results = model(img, size=640)  # reduce size=320 for faster inference
        return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    #parser.add_argument("--port", default=5000, type=int, help="port number")
    #args = parser.parse_args()
    port = int(os.environ.get('PORT',5000))

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)  # force_reload to recache
    #app.run(host="0.0.0.0", threaded=True, port=args.port)  # debug=True causes Restarting with stat
    app.run(host="0.0.0.0", threaded=True, port=port)
