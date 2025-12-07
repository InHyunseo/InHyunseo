import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from enum import IntEnum
import math

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

    def calibrate(self, kps):
        """ 사용자 신체 사이즈 측정 및 고스트 초기화 """
        KP = Yoloidx
        
        # 두 점 사이 거리 계산 헬퍼
        def d(i1, i2): 
            return np.linalg.norm(kps[i1] - kps[i2])

        # 신체 부위별 길이 측정
        self.limb_lengths = {
            "thigh": d(KP.R_HIP, KP.R_KNEE),
            "shin": d(KP.R_KNEE, KP.R_ANKLE),
            "torso": d(KP.R_SHOULDER, KP.R_HIP)
        }
        
        # 고스트의 기준점은 사용자의 발목 위치로 설정
        self.anchor_point = tuple(kps[KP.R_ANKLE].astype(int))
        
        # 상태 변경
        self.state = "READY"
        self.start_time = time.time()
        print(f"Calibration Done! Limbs: {self.limb_lengths}")

    def get_ghost_pose(self, t_cycle):
        """ 현재 사이클 시간(t_cycle)에 맞는 고스트 좌표 반환 """
        progress = 0.0
        
        # [Phase 1] Down (2.5초)
        if t_cycle < self.T_DOWN:
            progress = (t_cycle / self.T_DOWN) * 0.5
        # [Phase 2] Hold (0.5초)
        elif t_cycle < (self.T_DOWN + self.T_HOLD):
            progress = 0.5
        # [Phase 3] Up (1.5초)
        else:
            t_up = t_cycle - (self.T_DOWN + self.T_HOLD)
            progress = 0.5 + (t_up / self.T_UP) * 0.5

        # Easing (부드러운 움직임)
        if progress <= 0.5: 
            t = progress / 0.5
        else: 
            t = (1.0 - progress) / 0.5
        
        ratio = math.sin(t * math.pi / 2)

        # 각도 보간 (Interpolation)
        ang_shin = np.interp(ratio, [0, 1], [280, 330])
        ang_thigh = np.interp(ratio, [0, 1], [260, 175])
        ang_torso = np.interp(ratio, [0, 1], [275, 320])

        # 극좌표 -> 직교좌표 변환
        def p2c(rho, deg):
            rad = math.radians(deg)
            return rho * math.cos(rad), rho * math.sin(rad)
        
        # 좌표 계산 Chain
        ax, ay = self.anchor_point
        dx, dy = p2c(self.limb_lengths["shin"], ang_shin)
        kx, ky = ax + dx, ay + dy
        dx, dy = p2c(self.limb_lengths["thigh"], ang_thigh)
        hx, hy = kx + dx, ky + dy
        dx, dy = p2c(self.limb_lengths["torso"], ang_torso)
        sx, sy = hx + dx, hy + dy
        
        return {
            "ankle": (int(ax), int(ay)), 
            "knee": (int(kx), int(ky)),
            "hip": (int(hx), int(hy)), 
            "shoulder": (int(sx), int(sy))
        }

    def update(self):
        """ 메인 루프에서 호출: 현재 상태와 고스트 포즈 리턴 """
        if self.state != "EXERCISE": 
            return 0, 0, {}
            
        elapsed = time.time() - self.start_time
        t_cycle = elapsed % self.T_TOTAL
        rep = int(elapsed // self.T_TOTAL)
        
        pose = self.get_ghost_pose(t_cycle)
        return t_cycle, rep, pose

# ==========================================
# 4. Main Logic
# ==========================================
def main():
    print("Loading YOLO model...")
    # Jetson에서는 engine 파일 사용 권장 ('yolov8n-pose.engine')
    model = YOLO('yolov8n-pose.pt') 

    try:
        # GStreamer 파이프라인 시도
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    except:
        # 실패 시 웹캠
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not found")
        return

    ghost = GhostSystem()
    print("System Started. Please stand in position.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # YOLO 추론 (ID 트래킹 활성화)
        results = model.track(frame, persist=True, verbose=False, conf=0.5)
        
        if results[0].keypoints is not None and results[0].boxes is not None:
            # GPU 텐서를 CPU numpy로 변환
            kp_tensor = results[0].keypoints.xy
            box_tensor = results[0].boxes.xyxy
            ids_tensor = results[0].boxes.id
            
            if len(kp_tensor) > 0:
                # 첫 번째 사람만 처리 (Multi-user 확장 가능)
                kps = kp_tensor[0].cpu().numpy()
                bbox = box_tensor[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                # --- State Machine ---
                
                # [STATE 1] Calibration
                if ghost.state == "CALIB":
                    # 발목과 엉덩이가 보이면 측정
                    if kps[Yoloidx.R_ANKLE][1] > 0 and kps[Yoloidx.R_HIP][1] > 0:
                        cv2.putText(frame, "MEASURING...", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        ghost.calibrate(kps)
                
                # [STATE 2] Ready (Countdown)
                elif ghost.state == "READY":
                    elapsed = time.time() - ghost.start_time
                    remain = 3.0 - elapsed
                    
                    if remain <= 0:
                        ghost.state = "EXERCISE"
                        ghost.start_time = time.time() # 운동 시작 시점으로 리셋
                    else:
                        # 카운트다운 표시
                        cv2.putText(frame, f"{int(remain) + 1}", (x1 + 50, y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
                        # 대기 중 앵커 포인트 지속 업데이트 (발 움직임 반영)
                        ghost.anchor_point = tuple(kps[Yoloidx.R_ANKLE].astype(int))

                # [STATE 3] Exercise Loop
                elif ghost.state == "EXERCISE":
                    t_cycle, rep, ghost_pose = ghost.update()
                    
                    # ----------------------------------
                    # 1. Ghost Drawing (Skeleton)
                    # ----------------------------------
                    pts = list(ghost_pose.values())
                    # 라인 그리기
                    for i in range(len(pts)-1):
                        cv2.line(frame, pts[i], pts[i+1], (0, 255, 255), 6) # 노란색 굵은 선
                    # 관절 점 그리기
                    for p in pts:
                        cv2.circle(frame, p, 8, (0, 100, 255), -1)

                    # ----------------------------------
                    # 2. Bounding Box
                    # ----------------------------------
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # ----------------------------------
                    # 3. Status Logic (Color & Text)
                    # ----------------------------------
                    state_text = ""
                    state_color = (0, 0, 0)
                    
                    if t_cycle < ghost.T_DOWN:
                        state_text = "DOWN"
                        state_color = (0, 255, 255) # Yellow
                    elif t_cycle < (ghost.T_DOWN + ghost.T_HOLD):
                        state_text = "HOLD"
                        state_color = (0, 0, 255)   # Red (강조)
                    else:
                        state_text = "UP"
                        state_color = (255, 100, 0) # Blue/Orange

                    # ----------------------------------
                    # 4. UI: Top (Rep Count)
                    # ----------------------------------
                    rep_label = f"Rep: {rep}"
                    (tw, th), _ = cv2.getTextSize(rep_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    
                    # 머리 위 검은 박스
                    cv2.rectangle(frame, (x1, y1 - 40), (x1 + tw + 20, y1), (0, 0, 0), -1)
                    cv2.putText(frame, rep_label, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # ----------------------------------
                    # 5. UI: Bottom (Current State)
                    # ----------------------------------
                    state_label = f"{state_text}"
                    (tw, th), _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    
                    # 발 아래 컬러 박스
                    cv2.rectangle(frame, (x1, y2), (x1 + tw + 20, y2 + 40), state_color, -1)
                    
                    # HOLD일 때는 흰 글씨, 나머지는 검은 글씨 (가독성)
                    text_color = (255, 255, 255) if state_text == "HOLD" else (0, 0, 0)
                    cv2.putText(frame, state_label, (x1 + 10, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

                    # ----------------------------------
                    # 6. UI: Right (Depth Gauge)
                    # ----------------------------------
                    bar_w = 20
                    bar_x = x2 + 10 # 박스 오른쪽 옆
                    bar_h = (y2 - y1) # 박스 높이만큼
                    
                    # 게이지 배경 (회색)
                    cv2.rectangle(frame, (bar_x, y1), (bar_x + bar_w, y2), (50, 50, 50), -1)
                    
                    # 각도 계산 및 비율 매핑
                    hip_angle, _ = get_angle_3d(kps)
                    # 170도(서있음) ~ 80도(앉음) 기준으로 0~1 매핑
                    fill_ratio = (170 - hip_angle) / (170 - 80)
                    fill_ratio = np.clip(fill_ratio, 0.0, 1.0)
                    
                    # 게이지 채우기 높이
                    fill_h = int(bar_h * fill_ratio)
                    
                    # 색상 그라데이션 (빨강 -> 초록)
                    # 깊게 앉을수록(비율이 높을수록) 초록색
                    bar_color = (0, 255, int(255 * (1 - fill_ratio)))
                    
                    # 위에서 아래로 채워짐 (Depth 느낌)
                    cv2.rectangle(frame, (bar_x, y1), (bar_x + bar_w, y1 + fill_h), bar_color, -1)
                    
                    # 목표 지점 라인 (100% 지점)
                    target_y = y1 + int(bar_h * 1.0)
                    cv2.line(frame, (bar_x - 5, target_y), (bar_x + bar_w + 5, target_y), (0, 0, 255), 2)
        
        # 화면 표시
        cv2.imshow("Ghost Squat Trainer", frame)
        
        # ESC 키 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()