import cv2
import numpy as np
import mediapipe as mp
import math
import time

# ========= 1) 설정 =========
CAM_INDEX = 0
TARGET_KNEE_MIN = 70
TARGET_KNEE_MAX = 100
STAND_ANGLE_TH = 160

# ========= 2) 유틸 함수 =========
def calc_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    ab = np.array([ax - bx, ay - by])
    cb = np.array([cx - bx, cy - by])

    dot = np.dot(ab, cb)
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    if norm_ab == 0 or norm_cb == 0:
        return None

    cos_angle = dot / (norm_ab * norm_cb)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))
    return angle

def draw_text(img, text, org, scale=0.8, color=(0, 255, 0), thickness=2):
    cv2.putText(
        img, text, org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )

# ========= 3) MediaPipe Pose =========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========= 4) 카메라 =========
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("카메라 오류: CAM_INDEX를 바꿔보세요.")
    exit(1)

print("[i] Press 'q' to quit")
print("[i] Stand sideways and start squatting!")

# 상태 변수
squat_count = 0
state = "STAND"
prev_time = time.time()

# ========= 피드백 스무딩을 위한 변수 =========
last_feedback = ""
last_feedback_time = 0
FEEDBACK_HOLD = 2.0  # 초 동안 유지


# ========= 메인 루프 =========
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    knee_angle = None
    hip_angle = None
    current_feedback = ""

    # ========= 관절 추출 =========
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        r_hip      = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        r_knee     = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
        r_ankle    = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

        s = (int(r_shoulder.x * w), int(r_shoulder.y * h))
        hpt = (int(r_hip.x * w), int(r_hip.y * h))
        k = (int(r_knee.x * w), int(r_knee.y * h))
        a = (int(r_ankle.x * w), int(r_ankle.y * h))

        # 각도 계산
        knee_angle = calc_angle(hpt, k, a)
        hip_angle = calc_angle(s, hpt, k)

        # 시각화
        cv2.circle(frame, s, 6, (255, 255, 0), -1)
        cv2.circle(frame, hpt, 6, (0, 255, 255), -1)
        cv2.circle(frame, k, 6, (0, 255, 0), -1)
        cv2.circle(frame, a, 6, (0, 128, 255), -1)

        cv2.line(frame, s, hpt, (200, 200, 200), 2)
        cv2.line(frame, hpt, k, (200, 200, 200), 2)
        cv2.line(frame, k, a, (200, 200, 200), 2)

        if knee_angle:
            draw_text(frame, f"Knee: {knee_angle:5.1f}", (30, 80))
        if hip_angle:
            draw_text(frame, f"Hip:  {hip_angle:5.1f}", (30, 110))

        # ========= 상태 머신 =========
        if knee_angle:
            if state == "STAND":
                if knee_angle < STAND_ANGLE_TH:
                    state = "DOWN"

            elif state == "DOWN":
                if TARGET_KNEE_MIN <= knee_angle <= TARGET_KNEE_MAX:
                    state = "BOTTOM"
                elif knee_angle > STAND_ANGLE_TH:
                    state = "STAND"

            elif state == "BOTTOM":
                if knee_angle < TARGET_KNEE_MIN - 10:
                    current_feedback = "⚠ Too deep!"
                if knee_angle > TARGET_KNEE_MAX:
                    state = "UP"

            elif state == "UP":
                if knee_angle > STAND_ANGLE_TH:
                    squat_count += 1
                    state = "STAND"

            # ========= 피드백 조건 =========
            if state in ["DOWN", "BOTTOM"]:
                if knee_angle > TARGET_KNEE_MAX + 10:
                    current_feedback = "Go lower."
                elif TARGET_KNEE_MIN <= knee_angle <= TARGET_KNEE_MAX:
                    current_feedback = "✅ Excellent!"
                elif knee_angle < TARGET_KNEE_MIN - 10:
                    current_feedback = "⚠ Too deep!"

            if hip_angle:
                if hip_angle < 90:
                    current_feedback = "⚠ Leaning forward too much."
                elif 140 <= hip_angle <= 180 and state in ["DOWN", "BOTTOM"]:
                    current_feedback = "Good form."

    # ========= 피드백 스무딩 로직 =========
    if current_feedback:
        if current_feedback != last_feedback:
            last_feedback = current_feedback
            last_feedback_time = time.time()

    # 유지시간 지나면 메시지 제거
    if time.time() - last_feedback_time > FEEDBACK_HOLD:
        last_feedback = ""

    # ========= HUD =========
    cur_time = time.time()
    dt = max(1e-3, cur_time - prev_time)
    prev_time = cur_time
    fps = 1.0 / dt

    draw_text(frame, f"Squat Count: {squat_count}", (30, 40), scale=1.0)
    draw_text(frame, f"State: {state}", (30, 150), color=(255, 255, 0))
    draw_text(frame, f"FPS: {fps:4.1f}", (30, 180), scale=0.6, color=(200, 200, 200))

    # ========= 스무딩된 피드백 출력 =========
    if last_feedback:
        draw_text(frame, last_feedback, (30, 240), color=(0, 200, 255))

    cv2.imshow("Squat Feedback (Smooth)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
