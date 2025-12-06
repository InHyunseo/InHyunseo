import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from enum import IntEnum

# ==========================================
# 1. Yolo config & mapping
# ==========================================
class Yoloidx(IntEnum):
    NOSE = 0
    L_EYE = 1; R_EYE = 2
    L_EAR = 3; R_EAR = 4
    L_SHOULDER = 5; R_SHOULDER = 6
    L_ELBOW = 7; R_ELBOW = 8
    L_WRIST = 9; R_WRIST = 10
    L_HIP = 11; R_HIP = 12
    L_KNEE = 13; R_KNEE = 14
    L_ANKLE = 15; R_ANKLE = 16


def gstreamer_pipeline(sensor_id=0, capture_width=1920, capture_height=1080, display_width=1280, display_height=720, framerate=30, flip_method=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (sensor_id, capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )

# ==========================================
# 2. helper functions
# ==========================================
def calc_angle_3d(a, b, c):
    """
    Calculate the angle (in degrees) formed by three points a-b-c in 3D space, with b as the vertex.
    """
    v1 = a-b
    v2 = c-b

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == norm2 == 0:
        return 0.0
    
    cosine_ang = np.dot(v1, v2) / (norm1 * norm2)
    angle = np.degrees(np.arccos(np.clip(cosine_ang, -1.0, 1.0)))
    return angle


def get_angle_3d(keypoints_np):
    """
    keypoints_np: (17, 2) 형태의 좌표 배열
    """
    KP = Yoloidx
    
    # [수정] Enum 인덱스가 아니라, 실제 좌표 배열을 넘겨줘야 함
    hip_ang_r = calc_angle_3d(keypoints_np[KP.R_SHOULDER], keypoints_np[KP.R_HIP], keypoints_np[KP.R_KNEE])
    hip_ang_l = calc_angle_3d(keypoints_np[KP.L_SHOULDER], keypoints_np[KP.L_HIP], keypoints_np[KP.L_KNEE])
    hip_ang = 0.5 * (hip_ang_r + hip_ang_l)

    knee_ang_r = calc_angle_3d(keypoints_np[KP.R_HIP], keypoints_np[KP.R_KNEE], keypoints_np[KP.R_ANKLE])
    knee_ang_l = calc_angle_3d(keypoints_np[KP.L_HIP], keypoints_np[KP.L_KNEE], keypoints_np[KP.L_ANKLE])
    knee_ang = 0.5 * (knee_ang_r + knee_ang_l)
    
    return hip_ang, knee_ang

class GhostSystem:
    def __init__(self):
        self.state = "CALIB" # CALIB -> READY -> EXERCISE
        self.limb_lengths = {}
        self.anchor_point = (0, 0)
        self.start_time = None
        
        # 타이밍 설정 (Down 2.5s -> Hold 0.5s -> Up 1.5s)
        self.T_DOWN = 2.5
        self.T_HOLD = 0.5
        self.T_UP = 1.5
        self.T_TOTAL = self.T_DOWN + self.T_HOLD + self.T_UP

    def calibrate