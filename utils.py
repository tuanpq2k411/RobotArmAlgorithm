# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to display the pose detection results."""

import cv2
import numpy as np
from tflite_support.task import processor
import math
import time

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red
# Horizontal size of bottle
obj_size = 0.047
# vertical and horizontal view angle of camera
ver_view_angle = 28
hor_view_angle = 37
# position and angle of camera relative to robot arm
cam_pos_robot = (-0.07, -0.04, 0.21)
cam_angle_robot = 30

L1 = 120
L2 = 120
L3 = 90
p1 = (0, 0)
p2 = (0, 0)
p3 = (0, 0)
a1 = 90
a2 = 90
a3 = 90





def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.

    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.

    Returns:
      Image with bounding boxes.
    """
    A1 = A2 = A3 =0
    success = False
    for detection in detection_result.detections:
        if detection.categories[0].score > 0.75:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

            # Calculate scale ratio between bottle and frame
            scale = 640 / (end_point[0] - start_point[0])
            # Size of frame in real life
            img_hor_size = obj_size * scale
            img_ver_size = img_hor_size * 3 / 4
            # distance from camera to frame in real life
            distance_cam = img_hor_size / 2 / \
                math.tan(hor_view_angle / 2 / 180*math.pi)
            # object's position relative to camera
            object_y = -((end_point[0]+start_point[0])/2-320)/640 * img_hor_size
            object_z = -((end_point[1]+start_point[1])/2-240)/480 * img_ver_size
            object_x = distance_cam
            object_pos_robot = convert_cam_to_robot((object_x, object_y, object_z))
            print("object position: x=", object_pos_robot[0], "y=", object_pos_robot[1], "z=",object_pos_robot[2])
            # print("x", object_x, "y", object_y, "z", object_z)
            
            if(abs(object_pos_robot[1] < 0.07) and bbox.height > bbox.width*1.4):
                try:
                    A1,A2,A3 = find_angle((object_pos_robot[0]*1000,object_pos_robot[2]*1000))
                    success = True
                except:
                    success = False

            elif (abs(object_pos_robot[1] < 0.07) and bbox.height < bbox.width*1.4):
                try:
                    A1,A2,A3 = find_angle((object_pos_robot[0]*1000,object_pos_robot[2]*1000 - 50))
                    success = True
                except: 
                    success = False

            # Draw label and score
            # category = detection.categories[0]
            # category_name = category.category_name
            # probability = round(category.score, 2)
            # result_text = category_name + ' (' + str(probability) + ')'
            # text_location = (_MARGIN + bbox.origin_x,
            #                  _MARGIN + _ROW_SIZE + bbox.origin_y)
            # cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
            #             _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
    return image, success, A1, A2, A3


def convert_cam_to_robot(obj_pos_cam):
    # position in y axis of object cam = cam real
    y_cam_real = obj_pos_cam[1]
    x_cam = obj_pos_cam[0]
    z_cam = obj_pos_cam[2]
    d_cam = math.sqrt(x_cam**2 + z_cam**2)
    alpha_cam = math.atan2(x_cam, z_cam)
    alpha_cam_real = alpha_cam + cam_angle_robot/180*math.pi
    x_cam_real = d_cam * math.sin(alpha_cam_real)
    z_cam_real = d_cam * math.cos(alpha_cam_real)
    # calculate object's position relative to robot arm, add vector cam_real and cam position relative to robot
    x_robot = x_cam_real + cam_pos_robot[0]
    y_robot = y_cam_real + cam_pos_robot[1]
    z_robot = z_cam_real + cam_pos_robot[2]
    return (x_robot, y_robot, z_robot)

def find_angle(target, learning_rate=0.0001, max_iter=5000, tol=0.5):
    """
    Inverse kinematics using Gradient Descent.

    Given a target (x, y) in mm, compute joint angles (in degrees) for 3 servos.
    Forward kinematics (which is also the foundation of the loss function):
        phi1 = a1
        phi2 = a1 + a2
        phi3 = a1 + a2 + a3
        x = L1*cos(phi1) + L2*cos(phi2) + L3*cos(phi3)
        y = L1*sin(phi1) + L2*sin(phi2) + L3*sin(phi3)
    Loss:
        E = 0.5 * ((x - x_target)^2 + (y - y_target)^2)
    """
    x_d, y_d = target

    # Initial joint angles in radians
    a1 = math.pi / 4
    a2 = 0.0
    a3 = 0.0

    for _ in range(max_iter):
        # --- Forward Kinematics (= core of the loss function) ---
        phi1 = a1
        phi2 = a1 + a2
        phi3 = a1 + a2 + a3

        x = L1 * math.cos(phi1) + L2 * math.cos(phi2) + L3 * math.cos(phi3)
        y = L1 * math.sin(phi1) + L2 * math.sin(phi2) + L3 * math.sin(phi3)

        err_x = x - x_d
        err_y = y - y_d

        # --- Loss E = 0.5 * ((x - x_d)^2 + (y - y_d)^2) ---
        loss = 0.5 * (err_x ** 2 + err_y ** 2)

        # Check convergence
        if loss < 0.5 * tol ** 2:
            break

        # --- Gradients of loss w.r.t. joint angles ---
        de_da1 = (err_x * (-L1 * math.sin(phi1) - L2 * math.sin(phi2) - L3 * math.sin(phi3))
                  + err_y * ( L1 * math.cos(phi1) + L2 * math.cos(phi2) + L3 * math.cos(phi3)))

        de_da2 = (err_x * (-L2 * math.sin(phi2) - L3 * math.sin(phi3))
                  + err_y * ( L2 * math.cos(phi2) + L3 * math.cos(phi3)))

        de_da3 = (err_x * (-L3 * math.sin(phi3))
                  + err_y * ( L3 * math.cos(phi3)))

        # --- Gradient Descent update ---
        a1 -= learning_rate * de_da1
        a2 -= learning_rate * de_da2
        a3 -= learning_rate * de_da3

    # --- Map joint angles to servo angles (degrees) ---
    A1 = math.degrees(a1) + 90
    A2 = math.degrees(a2) + 90
    A3 = math.degrees(a3) + 90

    return A1, A2, A3

