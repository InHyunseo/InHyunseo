import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
from enum import IntEnum
# ==========================================
# 1. helper functions
# ==========================================
def calc_angle_3d(a, b, c):
    """
    Calculate the angle (in degrees) formed by three points a-b-c in 3D space, with b as the vertex.
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
    # for link with mediapipe functions (I made this first for mediapipe)
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

def draw_ghost_squat(frame, keypoints, idx_map, hip_center_x, hip_center_y, alpha=0.6):
    """
    사용자의 신체 사이즈를 기반으로 '이상적인 스쿼트 자세(Ghost)'를 오버레이합니다.
    frame: 원본 이미지
    keypoints: 현재 사용자의 키포인트 좌표
    idx_map: 인덱스 매핑 (YoloIndices)
    hip_center: 사용자의 엉덩이 중심 좌표 (기준점)
    alpha: 투명도 (0.0 ~ 1.0)
    """
    overlay = frame.copy()
    
    # 1. 신체 사이즈 측정 (현재 사용자의 픽셀 단위 길이 계산)
    # 오른쪽 다리 기준으로 길이 측정 (측면 뷰 가정)
    r_hip = keypoints[idx_map.RIGHT_HIP][:2]
    r_knee = keypoints[idx_map.RIGHT_KNEE][:2]
    r_ankle = keypoints[idx_map.RIGHT_ANKLE][:2]
    r_shoulder = keypoints[idx_map.RIGHT_SHOULDER][:2]

    # 허벅지 길이 (Hip -> Knee)
    thigh_len = np.linalg.norm(r_hip - r_knee)
    # 정강이 길이 (Knee -> Ankle)
    shin_len = np.linalg.norm(r_knee - r_ankle)
    # 몸통 길이 (Shoulder -> Hip)
    torso_len = np.linalg.norm(r_shoulder - r_hip)

    # 2. 이상적인 스쿼트 좌표 계산 (Side View 기준)
    # 기준점: 사용자의 현재 엉덩이 위치 (hip_center_x, hip_center_y)
    
    # [Ghost] 무릎 위치: 엉덩이에서 앞으로 뻗어나감 (이상적인 깊이)
    # 스쿼트 정석: 허벅지가 바닥과 수평(0도) 또는 살짝 아래
    ghost_knee_x = hip_center_x + thigh_len  # 측면이니까 앞으로(X축 증가)
    ghost_knee_y = hip_center_y              # 수평이니까 Y축 동일 (조금 내리려면 + 값)

    # [Ghost] 발목 위치: 무릎에서 수직으로 아래로
    # 스쿼트 정석: 정강이는 바닥과 수직에 가깝거나 발목이 무릎보다 뒤에 있지 않음
    ghost_ankle_x = ghost_knee_x - (shin_len * 0.2) # 무릎보다 살짝 뒤 (자연스러운 각도)
    ghost_ankle_y = ghost_knee_y + shin_len

    # [Ghost] 어깨 위치: 엉덩이 위로 꼿꼿하게 (허리 펴기)
    # 상체 각도: 앞으로 너무 숙이지 않게 (약 60~70도 유지)
    lean_forward = torso_len * 0.3 # 상체가 앞으로 약간 기우는 정도
    ghost_shoulder_x = hip_center_x + lean_forward 
    ghost_shoulder_y = hip_center_y - (torso_len * 0.9) # 위로 올라감

    # 3. 그리기 (반투명 선)
    # 좌표들을 정수형(int)으로 변환
    pt_hip = (int(hip_center_x), int(hip_center_y))
    pt_knee = (int(ghost_knee_x), int(ghost_knee_y))
    pt_ankle = (int(ghost_ankle_x), int(ghost_ankle_y))
    pt_shoulder = (int(ghost_shoulder_x), int(ghost_shoulder_y))

    # Ghost 색상 (형광 녹색)
    ghost_color = (0, 255, 0) 
    thickness = 5

    # 몸통 (Hip -> Shoulder)
    cv2.line(overlay, pt_hip, pt_shoulder, ghost_color, thickness)
    # 허벅지 (Hip -> Knee)
    cv2.line(overlay, pt_hip, pt_knee, ghost_color, thickness)
    # 정강이 (Knee -> Ankle)
    cv2.line(overlay, pt_knee, pt_ankle, ghost_color, thickness)

    # 관절 포인트
    cv2.circle(overlay, pt_knee, 8, (255, 255, 255), -1)
    cv2.circle(overlay, pt_shoulder, 8, (255, 255, 255), -1)

    # 4. 이미지 합성 (투명도 적용)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


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

                # [수정된 부분 1] 고스트 오버레이 조건 변경
                # 기존: if user_data[track_id]['stage'] == 'down': (너무 늦게 뜸)
                # 변경: 서 있는 각도(160도)보다 조금이라도 굽히면 바로 표시
                if hip_angle < 165: 
                     draw_ghost_squat(
                        annotated_frame, keypoints, idx_map, hip_x, hip_y, 
                        alpha=0.5
                    )
                     # 가이드 문구
                     cv2.putText(annotated_frame, "Match Green Line!", (hip_x - 60, hip_y - 80),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # [수정된 부분 2] 텍스트 가독성 개선
                info_text = f"ID:{track_id} CNT:{user_data[track_id]['count']}"
                
                # 배경 박스 (랜덤 컬러)
                cv2.rectangle(annotated_frame, (hip_x - 10, hip_y - 40), (hip_x + 150, hip_y), user_data[track_id]['color'], -1)
                
                # 글씨 색을 검은색(0, 0, 0)으로 변경 -> 흰색 박스에서도 잘 보임!
                cv2.putText(annotated_frame, info_text, (hip_x, hip_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # <--- 여기가 변경됨

            cv2.imshow("Multi-Person Squat", annotated_frame)
        else:
            cv2.imshow("Multi-Person Squat", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()