import cv2
import time
import math
import numpy as np

# ==========================================
# 1. 설정 (Config)
# ==========================================
LIMB_LENGTHS = {
    "thigh": 140, "shin": 130, "torso": 160
}
ANCHOR_POINT = (400, 400) 

# [수정된 타이밍]
TIME_DOWN = 2.5   # 천천히 내려가기 (자극 집중)
TIME_HOLD = 0.5   # 바닥 정지 (반동 제어)
TIME_UP   = 1.5   # 빠르게 올라오기 (파워)
TOTAL_CYCLE = TIME_DOWN + TIME_HOLD + TIME_UP

def get_ghost_pose(progress, anchor, lengths):
    """
    progress (0.0 ~ 1.0)에 따라 관절 좌표를 계산
    """
    # Easing Logic (움직임 곡선)
    if progress <= 0.5:
        t = progress / 0.5 
    else:
        t = (1.0 - progress) / 0.5 

    ratio = math.sin(t * math.pi / 2)

    # 상체 각도 수정 버전 (앞으로 숙임)
    ang_shin = np.interp(ratio, [0, 1], [280, 330])
    ang_thigh = np.interp(ratio, [0, 1], [260, 175])
    ang_torso = np.interp(ratio, [0, 1], [275, 320])

    def pol2cart(rho, phi_deg):
        phi_rad = math.radians(phi_deg)
        return rho * math.cos(phi_rad), rho * math.sin(phi_rad)

    ax, ay = anchor
    dx, dy = pol2cart(lengths["shin"], ang_shin)
    kx, ky = ax + dx, ay + dy
    dx, dy = pol2cart(lengths["thigh"], ang_thigh)
    hx, hy = kx + dx, ky + dy
    dx, dy = pol2cart(lengths["torso"], ang_torso)
    sx, sy = hx + dx, hy + dy

    return {
        "ankle": (int(ax), int(ay)),
        "knee": (int(kx), int(ky)),
        "hip": (int(hx), int(hy)),
        "shoulder": (int(sx), int(sy))
    }

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Ghost Squat: Down({TIME_DOWN}s) -> Hold({TIME_HOLD}s) -> Up({TIME_UP}s)")
    
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. 사이클 시간 계산
        elapsed = time.time() - start_time
        rep_count = int(elapsed // TOTAL_CYCLE)
        t_in_cycle = elapsed % TOTAL_CYCLE

        progress = 0.0
        state_text = ""
        state_color = (0, 255, 0)

        # [Phase 1] Down (0 ~ 2.5초)
        if t_in_cycle < TIME_DOWN:
            # 시간 비율(0~1)을 progress 절반(0~0.5)에 매핑
            ratio = t_in_cycle / TIME_DOWN
            progress = ratio * 0.5
            state_text = "DOWN"
            state_color = (0, 255, 255) # Yellow

        # [Phase 2] Hold (2.5 ~ 3.0초)
        elif t_in_cycle < (TIME_DOWN + TIME_HOLD):
            progress = 0.5 # 가장 깊은 자세 고정
            state_text = "HOLD"
            state_color = (0, 0, 255) # Red (강조)

        # [Phase 3] Up (3.0 ~ 4.5초)
        else:
            t_up_start = t_in_cycle - (TIME_DOWN + TIME_HOLD)
            ratio = t_up_start / TIME_UP
            # progress 절반(0.5)에서 끝(1.0)까지 매핑
            progress = 0.5 + (ratio * 0.5)
            state_text = "UP"
            state_color = (255, 100, 0) # Blue

        # 2. 그리기
        joints = get_ghost_pose(progress, ANCHOR_POINT, LIMB_LENGTHS)
        
        color = (0, 255, 255)
        thickness = 5
        pts = list(joints.values())
        
        for i in range(len(pts)-1):
            cv2.line(frame, pts[i], pts[i+1], color, thickness)
        for pt in pts:
            cv2.circle(frame, pt, 8, (255, 0, 0), -1)

        # 3. UI 표시
        # 배경 박스 (가독성 UP)
        cv2.rectangle(frame, (10, 10), (380, 80), (0, 0, 0), -1)
        
        # Reps
        cv2.putText(frame, f"Rep: {rep_count}", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # State
        cv2.putText(frame, state_text, (220, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3)
        
        # Progress Bar (하단에 시간 흐름 표시)
        bar_width = int((t_in_cycle / TOTAL_CYCLE) * 640)
        cv2.rectangle(frame, (0, 470), (bar_width, 480), state_color, -1)

        cv2.imshow("Ghost Debugger", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()