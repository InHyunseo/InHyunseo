import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
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

from enum import IntEnum  # <--- [중요] 이거 추가!

# ==========================================
# 2. YOLO -> Mediapipe 변환 어댑터 (수정됨)
# ==========================================
class LandmarkShim:
    """ Mediapipe의 landmark 객체(.x, .y, .z)를 흉내내는 클래스 """
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

# [수정] 일반 class 대신 IntEnum을 상속받도록 변경
# 이렇게 하면 정수(int)처럼 배열 인덱싱도 되고, .value 속성도 가집니다.
class YoloIndices(IntEnum):
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
# 3. 메인 실행 코드 (멀티 유저 지원)
# ==========================================
def main():
    print("Loading YOLO model...")
    model = YOLO('yolov8n-pose.pt') 
    
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Camera open failed!")
        return

    mp_pose_shim = MpPoseShim()

    # --- [멀티 유저 데이터 저장소] ---
    # ID를 키(Key)로 사용하여 각 사람의 상태를 따로 관리합니다.
    # 예: { 1: {'count': 0, 'stage': 'up'}, 2: {'count': 5, 'stage': 'down'} }
    user_data = {}

    STAND_THRESH = 160
    SQUAT_THRESH = 90

    print("✅ Multi-Person Squat Counter Started...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # [핵심 변경 1] predict -> track (persist=True 필수)
        # persist=True: 이전 프레임의 사람 ID를 기억해서 다음 프레임에도 유지함
        results = model.track(frame, persist=True, verbose=False, conf=0.5)

        if results[0].keypoints is not None and results[0].boxes.id is not None:
            # 감지된 모든 사람의 데이터를 가져옵니다.
            # boxes.id: 트래킹 ID 목록 (예: [1.0, 2.0])
            # keypoints.data: 관절 좌표 목록
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints_list = results[0].keypoints.data.cpu().numpy()

            # 시각화용 이미지
            annotated_frame = results[0].plot()

            # 감지된 각 사람에 대해 반복문 실행
            for track_id, keypoints in zip(track_ids, keypoints_list):
                
                # 처음 보는 ID라면 데이터 초기화
                if track_id not in user_data:
                    user_data[track_id] = {'count': 0, 'stage': 'up', 'color': np.random.randint(0, 255, 3).tolist()}

                # --- 1. 좌표 변환 (기존 로직) ---
                world_lms_shim = [LandmarkShim(0, 0, 0)] * 33
                idx_map = YoloIndices
                target_indices = [
                    idx_map.LEFT_SHOULDER, idx_map.RIGHT_SHOULDER,
                    idx_map.LEFT_HIP, idx_map.RIGHT_HIP,
                    idx_map.LEFT_KNEE, idx_map.RIGHT_KNEE,
                    idx_map.LEFT_ANKLE, idx_map.RIGHT_ANKLE
                ]
                
                # 현재 사람의 좌표 추출
                for idx in target_indices:
                    x, y = keypoints[idx][0], keypoints[idx][1]
                    world_lms_shim[idx] = LandmarkShim(x, y, 0.0)

                # --- 2. 각도 계산 ---
                hip_angle = get_hip_angle_3d(world_lms_shim, mp_pose_shim)

                # --- 3. 개별 카운팅 로직 ---
                curr_stage = user_data[track_id]['stage']
                
                if hip_angle > STAND_THRESH:
                    user_data[track_id]['stage'] = "up"
                
                if hip_angle < SQUAT_THRESH and curr_stage == "up":
                    user_data[track_id]['stage'] = "down"
                
                if hip_angle > STAND_THRESH and curr_stage == "down":
                    user_data[track_id]['stage'] = "up"
                    user_data[track_id]['count'] += 1
                    print(f"User {track_id}: Count {user_data[track_id]['count']}")

                # --- 4. 개별 시각화 (머리 위에 정보 띄우기) ---
                # 엉덩이 좌표를 기준으로 텍스트 표시
                hip_x = int(keypoints[idx_map.RIGHT_HIP][0])
                hip_y = int(keypoints[idx_map.RIGHT_HIP][1])

                info_text = f"ID:{track_id} CNT:{user_data[track_id]['count']}"
                
                # 텍스트 배경 박스
                cv2.rectangle(annotated_frame, (hip_x - 10, hip_y - 40), (hip_x + 150, hip_y), user_data[track_id]['color'], -1)
                cv2.putText(annotated_frame, info_text, (hip_x, hip_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Multi-Person Squat", annotated_frame)
        else:
            cv2.imshow("Multi-Person Squat", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()