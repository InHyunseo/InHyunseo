import cv2
print(cv2.getBuildInformation())
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# =========================
# 각도 계산용 헬퍼
# =========================
def calc_angle_3d(a, b, c):
    """
    3D 상에서 점 b를 중심으로, a-b-c 세 점이 이루는 각도(도 단위)
    a, b, c: world_landmark (x, y, z)
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


def get_hip_angle_3d(world_lms, mp_pose):
    """
    허리각: shoulder-hip-knee (좌/우 평균)
    """
    IDX_R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    IDX_L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    IDX_R_HP = mp_pose.PoseLandmark.RIGHT_HIP.value
    IDX_L_HP = mp_pose.PoseLandmark.LEFT_HIP.value
    IDX_R_KN = mp_pose.PoseLandmark.RIGHT_KNEE.value
    IDX_L_KN = mp_pose.PoseLandmark.LEFT_KNEE.value

    ang_r = calc_angle_3d(
        world_lms[IDX_R_SH],
        world_lms[IDX_R_HP],
        world_lms[IDX_R_KN]
    )
    ang_l = calc_angle_3d(
        world_lms[IDX_L_SH],
        world_lms[IDX_L_HP],
        world_lms[IDX_L_KN]
    )
    return 0.5 * (ang_r + ang_l)


def get_knee_angle_3d(world_lms, mp_pose):
    """
    무릎각: hip-knee-ankle (좌/우 평균)
    """
    IDX_R_HP = mp_pose.PoseLandmark.RIGHT_HIP.value
    IDX_L_HP = mp_pose.PoseLandmark.LEFT_HIP.value
    IDX_R_KN = mp_pose.PoseLandmark.RIGHT_KNEE.value
    IDX_L_KN = mp_pose.PoseLandmark.LEFT_KNEE.value
    IDX_R_AN = mp_pose.PoseLandmark.RIGHT_ANKLE.value
    IDX_L_AN = mp_pose.PoseLandmark.LEFT_ANKLE.value

    ang_r = calc_angle_3d(
        world_lms[IDX_R_HP],
        world_lms[IDX_R_KN],
        world_lms[IDX_R_AN]
    )
    ang_l = calc_angle_3d(
        world_lms[IDX_L_HP],
        world_lms[IDX_L_KN],
        world_lms[IDX_L_AN]
    )
    return 0.5 * (ang_r + ang_l)

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# =========================
# 스쿼트 파라미터
# =========================
# 타깃 바닥 자세 각도(요구사항)
HIP_ANGLE_TARGET  = 80.0
KNEE_ANGLE_TARGET = 60.0
HIP_ANGLE_TOL     = 10.0   # 허리 good 범위 ±10°
KNEE_ANGLE_TOL    = 10.0   # 무릎 good 범위 ±10°

CALIB_FRAMES = 60          # 서 있는 상태 칼리브레이션 프레임 수

# depth = 0: 서 있음, depth = 1: 이상적인 바닥
DEPTH_DOWN_TH    = 0.20    # 이 이상이면 내려가고 있다고 판단
DEPTH_BOTTOM_MIN = 0.80    # 이 이상이면 바닥(BOTTOM) 구간

# 속도(깊이 변화량/초)
VEL_MIN  = 0.4             # DOWN/UP 판정 최소 속도
VEL_ZERO = 0.15            # 거의 멈춘 상태

FEEDBACK_HOLD = 1.5        # 피드백 문구 유지 시간(초)


# =========================
# 상태 변수
# =========================
state = "CALIB"            # "CALIB", "STAND", "DOWN", "BOTTOM", "UP"
rep_count = 0

calibrated = False
calib_count = 0

base_hip_angle = 180.0
base_knee_angle = 180.0

hip_angle_smooth = None
knee_angle_smooth = None

prev_depth = 0.0
depth_smooth = 0.0
prev_time = time.time()
bottom_enter_time = time.time()

last_feedback = ""
last_feedback_time = 0.0


# =========================
# 웹캠 루프
# =========================
cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)

with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            continue

        # 성능 위해 writeable False → RGB 변환 후 추론
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape
        now = time.time()

        hip_angle = None
        knee_angle = None

        if results.pose_landmarks and results.pose_world_landmarks:
            world_lms = results.pose_world_landmarks.landmark

            # 3D 허리/무릎 각도 계산
            hip_angle = get_hip_angle_3d(world_lms, mp_pose)
            knee_angle = get_knee_angle_3d(world_lms, mp_pose)

            # 기본 스켈레톤(2D) 그리기
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # -------------------------
            # 1) CALIBRATION
            # -------------------------
            if not calibrated:
                calib_count += 1
                # 단순 이동평균
                alpha_c = 1.0 / calib_count
                base_hip_angle  = (1 - alpha_c) * base_hip_angle  + alpha_c * hip_angle
                base_knee_angle = (1 - alpha_c) * base_knee_angle + alpha_c * knee_angle

                text = f"CALIBRATING... ({calib_count}/{CALIB_FRAMES})"
                cv2.putText(image, text, (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(image,
                            f"Hip:{hip_angle:5.1f}  Knee:{knee_angle:5.1f}",
                            (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if calib_count >= CALIB_FRAMES:
                    calibrated = True
                    state = "STAND"
                    prev_time = now
                    prev_depth = 0.0
                    depth_smooth = 0.0
                    hip_angle_smooth = None
                    knee_angle_smooth = None
                    bottom_enter_time = now

                # 현재 프레임은 여기까지 그리고 다음 프레임으로
                cv2.imshow('Squat Tracker')
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                continue

            # -------------------------
            # 2) 각도 smoothing + depth 계산
            # -------------------------
            alpha = 0.25
            if hip_angle_smooth is None:
                hip_angle_smooth = hip_angle
                knee_angle_smooth = knee_angle
            else:
                hip_angle_smooth = (1 - alpha) * hip_angle_smooth + alpha * hip_angle
                knee_angle_smooth = (1 - alpha) * knee_angle_smooth + alpha * knee_angle

            ha = hip_angle_smooth
            ka = knee_angle_smooth

            # base(서 있을 때) → target(바닥)까지를 0~1로 정규화
            denom_hip  = max(5.0, base_hip_angle  - HIP_ANGLE_TARGET)
            denom_knee = max(5.0, base_knee_angle - KNEE_ANGLE_TARGET)

            depth_hip  = (base_hip_angle  - ha) / denom_hip
            depth_knee = (base_knee_angle - ka) / denom_knee

            depth_raw = 0.5 * (depth_hip + depth_knee)
            depth_raw = max(0.0, min(depth_raw, 1.5))

            dt = now - prev_time
            dt = max(dt, 1e-3)
            v = (depth_raw - prev_depth) / dt
            prev_depth = depth_raw
            prev_time = now

            if depth_smooth == 0.0:
                depth_smooth = depth_raw
            else:
                depth_smooth = (1 - alpha) * depth_smooth + alpha * depth_raw

            d = depth_smooth

            # -------------------------
            # 3) 상태머신
            # -------------------------
            feedback = ""

            if state == "STAND":
                if d > DEPTH_DOWN_TH and v > VEL_MIN:
                    state = "DOWN"

            elif state == "DOWN":
                if d >= DEPTH_BOTTOM_MIN and abs(v) < VEL_ZERO:
                    state = "BOTTOM"
                    bottom_enter_time = now
                elif d < DEPTH_DOWN_TH * 0.5 and abs(v) < VEL_ZERO:
                    state = "STAND"

            elif state == "BOTTOM":
                # 허리/무릎 각도 기반 깊이 평가
                err_hip  = ha - HIP_ANGLE_TARGET
                err_knee = ka - KNEE_ANGLE_TARGET

                if (abs(err_hip) <= HIP_ANGLE_TOL) and (abs(err_knee) <= KNEE_ANGLE_TOL):
                    feedback = "✅ Good depth"
                elif (err_hip > 0) or (err_knee > 0):
                    feedback = "Go deeper"
                else:
                    feedback = "⚠ Too deep, come up"

                if (now - bottom_enter_time) > 0.3 and v < -VEL_MIN:
                    state = "UP"

            elif state == "UP":
                if d < DEPTH_DOWN_TH and abs(v) < VEL_ZERO:
                    rep_count += 1
                    state = "STAND"
                    feedback = f"Rep {rep_count} done"

            # 피드백 유지 관리
            if feedback:
                last_feedback = feedback
                last_feedback_time = now
            else:
                if now - last_feedback_time > FEEDBACK_HOLD:
                    last_feedback = ""

            # -------------------------
            # HUD 출력
            # -------------------------
            cv2.putText(image,
                        f"State: {state}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image,
                        f"Reps: {rep_count}",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(image,
                        f"Hip: {ha:5.1f} deg  (target {HIP_ANGLE_TARGET:.0f})",
                        (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image,
                        f"Knee: {ka:5.1f} deg (target {KNEE_ANGLE_TARGET:.0f})",
                        (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(image,
                        f"Depth: {d:4.2f}",
                        (30, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if last_feedback:
                cv2.putText(image,
                            last_feedback,
                            (30, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        else:
            # 포즈 인식 안 될 때는 상태만 표시
            cv2.putText(image,
                        "No pose detected",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Squat Tracker')
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
