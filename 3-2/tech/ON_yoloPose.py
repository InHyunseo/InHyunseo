import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
print(torch.cuda.is_available())
# ==========================================
# 1. 기존 헬퍼 함수 (그대로 유지)
# ==========================================
def calc_angle_3d(a, b, c):
    """
    3D 상에서 점 b를 중심으로, a-b-c 세 점이 이루는 각도(도 단위)
    """
    pa = np.array([a.x, a.y, a.z])
    pb = np.array([b.x, b.y, b.z])
    pc = np.array([c.x, c.y, c.z])

    v1 = pa - pb
    v2 = pc - pb

    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cosang = np.dot(v1, v2) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def get_hip_angle_3d(world_lms, mp_pose_shim):
    # mp_pose 대신 아래에서 만든 Shim(가짜 객체)을 사용합니다.
    IDX_R_SH = mp_pose_shim.PoseLandmark.RIGHT_SHOULDER.value
    IDX_L_SH = mp_pose_shim.PoseLandmark.LEFT_SHOULDER.value
    IDX_R_HP = mp_pose_shim.PoseLandmark.RIGHT_HIP.value
    IDX_L_HP = mp_pose_shim.PoseLandmark.LEFT_HIP.value
    IDX_R_KN = mp_pose_shim.PoseLandmark.RIGHT_KNEE.value
    IDX_L_KN = mp_pose_shim.PoseLandmark.LEFT_KNEE.value

    ang_r = calc_angle_3d(world_lms[IDX_R_SH], world_lms[IDX_R_HP], world_lms[IDX_R_KN])
    ang_l = calc_angle_3d(world_lms[IDX_L_SH], world_lms[IDX_L_HP], world_lms[IDX_L_KN])
    return 0.5 * (ang_r + ang_l)

def get_knee_angle_3d(world_lms, mp_pose_shim):
    IDX_R_HP = mp_pose_shim.PoseLandmark.RIGHT_HIP.value
    IDX_L_HP = mp_pose_shim.PoseLandmark.LEFT_HIP.value
    IDX_R_KN = mp_pose_shim.PoseLandmark.RIGHT_KNEE.value
    IDX_L_KN = mp_pose_shim.PoseLandmark.LEFT_KNEE.value
    IDX_R_AN = mp_pose_shim.PoseLandmark.RIGHT_ANKLE.value
    IDX_L_AN = mp_pose_shim.PoseLandmark.LEFT_ANKLE.value

    ang_r = calc_angle_3d(world_lms[IDX_R_HP], world_lms[IDX_R_KN], world_lms[IDX_R_AN])
    ang_l = calc_angle_3d(world_lms[IDX_L_HP], world_lms[IDX_L_KN], world_lms[IDX_L_AN])
    return 0.5 * (ang_r + ang_l)

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=30, flip_method=0):
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
# 2. YOLO -> Mediapipe 변환 어댑터 (핵심!)
# ==========================================
class LandmarkShim:
    """ Mediapipe의 landmark 객체(.x, .y, .z)를 흉내내는 클래스 """
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z # YOLO는 기본적으로 Z가 없으므로 0으로 처리 (2D 각도 계산됨)

class YoloIndices:
    """ YOLO의 Keypoint 인덱스를 Mediapipe 이름으로 매핑 (COCO 포맷 기준) """
    # YOLO (COCO) Index Mapping
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

class MpPoseShim:
    """ mp_pose 모듈을 흉내내는 클래스 """
    PoseLandmark = YoloIndices

# ==========================================
# 3. 메인 실행 코드
# ==========================================
def main():
    # 모델 로드 (처음 실행시 다운로드됨)
    print("Loading YOLO model...")
    model = YOLO('yolov8n-pose.pt') 
    
    # GStreamer 파이프라인으로 카메라 열기
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Camera open failed!")
        return

    mp_pose_shim = MpPoseShim() # 기존 함수에 넘겨줄 가짜 mp_pose 객체

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO Inference (verbose=False로 로그 줄임)
        results = model.predict(frame, verbose=False, conf=0.5)

        # 사람이 감지되었는지 확인
        if results[0].keypoints is not None and results[0].keypoints.shape[1] > 0:
            # 첫 번째 사람의 키포인트만 가져옴 (17개 포인트)
            # data shape: (1, 17, 3) -> [x, y, conf] or [x, y] depending on version
            keypoints = results[0].keypoints.data[0].cpu().numpy()

            # YOLO 결과를 Mediapipe 포맷(List of objects)으로 변환
            # YOLO Keypoint는 0~16번까지 있음.
            # 하지만 Mediapipe 인덱스는 33번까지 있으므로, 안전하게 33개짜리 배열을 만들고 채워넣음
            world_lms_shim = [LandmarkShim(0, 0, 0)] * 33
            
            # 매핑된 중요 포인트만 업데이트
            idx_map = YoloIndices
            target_indices = [
                idx_map.LEFT_SHOULDER, idx_map.RIGHT_SHOULDER,
                idx_map.LEFT_HIP, idx_map.RIGHT_HIP,
                idx_map.LEFT_KNEE, idx_map.RIGHT_KNEE,
                idx_map.LEFT_ANKLE, idx_map.RIGHT_ANKLE
            ]

            for idx in target_indices:
                # keypoints[idx] = [x, y, confidence] (픽셀 좌표)
                x, y = keypoints[idx][0], keypoints[idx][1]
                # z는 0으로 둠 (2D 평면 각도로 계산하게 됨)
                world_lms_shim[idx] = LandmarkShim(x, y, 0.0)

            # --- 기존 로직 호출 ---
            hip_angle = get_hip_angle_3d(world_lms_shim, mp_pose_shim)
            knee_angle = get_knee_angle_3d(world_lms_shim, mp_pose_shim)

            # 시각화 (YOLO 내장 plot 사용하거나 직접 그림)
            annotated_frame = results[0].plot()
            
            # 정보 표시
            cv2.putText(annotated_frame, f"Hip: {hip_angle:.1f}", (30, 50), 
                        cv2.putText(annotated_frame, f"Knee: {knee_angle:.1f}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2))
            
            cv2.imshow("YOLO Pose Squat", annotated_frame)

        else:
            cv2.imshow("YOLO Pose Squat", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()