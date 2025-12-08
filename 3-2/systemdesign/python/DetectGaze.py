import cv2
import threading
import time
import random
import queue
import socket
import os
from ultralytics import YOLO

from gtts import gTTS
import pygame

# --- My recieving server (All)---
HOST = '0.0.0.0'
PORT = 5000

# 카메라 해상도 (C#에서 각도를 기반으로 좌표를 게산해서 넘겨주기 때문에 맞춰줘야함)
CAM_WIDTH = 640
CAM_HEIGHT = 480

# For gTTs flags, dwelling time settings
Dwelling_Threshold = 1.5 # 1.5초
dwell_target_name = None
dwell_time_start = 0.0
dwell_triggered = False 

# YOLO 설정
CONFIDENCE_THRESHOLD = 0.5
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)
# 라벨 오버레이 상태
LAST_LABEL_MSG = None
LAST_LABEL_TIME = 0.0
LABEL_MSG_DURATION = 1.5

# TTS 설정
TTS_COOLDOWN = 3.0  
tts_queue = queue.Queue()
last_spoken = {"msg": None, "t": 0.0}



def label_at_gaze(yolo_results, gx, gy, class_names):
    """ 시선점(gx, gy)이 포함된 박스 중 신뢰도가 가장 높은 라벨 반환 """
    if yolo_results is None:
        return None, None

    best = None
    best_conf = -1.0

    for data in yolo_results.boxes.data.tolist():
        xmin, ymin, xmax, ymax = map(int, data[:4])
        conf = float(data[4]); cls_id = int(data[5])
        
        margin = 10
        if (xmin - margin <= gx <= xmax + margin) and (ymin - margin <= gy <= ymax + margin):
            if conf > best_conf:
                best_conf = conf
                best = class_names[cls_id]

    return (best, best_conf) if best is not None else (None, None)

# --- 공유 변수 및 Lock ---
latest_frame = None
latest_yolo_results = None
latest_eog_gaze_data = {'x': CAM_WIDTH // 2, 'y': CAM_HEIGHT // 2}

frame_lock = threading.Lock()
yolo_results_lock = threading.Lock()
eog_gaze_lock = threading.Lock()

is_running = True

# --- 1. gTTS 스레드 ---
def tts_thread_func():
    # pygame 믹서 초기화 (오디오 재생용)
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"[오디오 초기화 실패] {e}")
        return

    print("🎤 Google TTS 시스템 준비 완료")

    while is_running:
        try:
            # 큐에서 메시지 대기
            msg = tts_queue.get(timeout=0.1)
            
            if msg:
                print(f"🔊 생성 중... : {msg}")
                
                # 1. 구글 서버에서 음성 파일 생성 (lang='ko' 한국어)
                tts = gTTS(text=msg, lang='en', slow=False)
                
                # 2. 임시 파일로 저장
                filename = "temp_voice.mp3"
                tts.save(filename)
                
                # 3. 재생
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                
                # 4. 재생이 끝날 때까지 대기
                while pygame.mixer.music.get_busy() and is_running:
                    time.sleep(0.1)
                
                # 5. 파일 연결 해제 (삭제를 위해)
                pygame.mixer.music.unload()
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[TTS 오류] {e}")
            # 인터넷 연결 문제일 수 있음

# --- 2. YOLO 스레드 ---
def yolo_thread_func(yolo_model):
    global latest_frame, latest_yolo_results, is_running
    print("YOLO 모델 가동 중...")

    while is_running:
        current_frame = None
        with frame_lock:
            if latest_frame is not None:
                current_frame = latest_frame.copy()

        if current_frame is not None:
            results = yolo_model(current_frame, imgsz=320, conf=CONFIDENCE_THRESHOLD, verbose=False)
            with yolo_results_lock:
                latest_yolo_results = results[0]
        
        time.sleep(0.01)

# --- 3. TCP 시선 데이터 수신 스레드 ---
def tcp_eog_thread_func():
    global latest_eog_gaze_data, is_running
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"🚀 시선 데이터 서버 대기중... ({HOST}:{PORT})")
        
        client_socket, addr = server_socket.accept()
        print(f"✅ C# 프로그램 연결됨: {addr}")
        
        buffer = ""
        
        while is_running:
            try:
                data = client_socket.recv(1024)
                if not data: break
                
                buffer += data.decode('utf-8')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line: continue
                    
                    if ',' in line:
                        try:
                            parts = line.split(',')
                            px = int(float(parts[0]))
                            py = int(float(parts[1]))
                            
                            px = max(0, min(CAM_WIDTH, px))
                            py = max(0, min(CAM_HEIGHT, py))
                            
                            with eog_gaze_lock:
                                latest_eog_gaze_data = {'x': px, 'y': py}
                                
                        except ValueError:
                            pass
            except Exception as e:
                print(f"데이터 수신 중 오류: {e}")
                break
                
    except Exception as e:
        print(f"TCP 서버 오류: {e}")
    finally:
        try:
            server_socket.close()
        except: pass
        print("TCP 서버 종료")

# --- 4. 시각화 (Dwell Progress) ---
def draw_dwell_ui(img, center, progress, triggered):
    radius = 20
    # 배경
    cv2.circle(img, center, radius, (200, 200, 200), 2)
    if triggered:
        # 완료 시 초록색 채움
        cv2.circle(img, center, radius, GREEN, -1)
    elif progress > 0:
        # 진행 중 노란색 호
        end_angle = -90 + (360 * progress)
        cv2.ellipse(img, center, (radius, radius), 0, -90, end_angle, YELLOW, 4)


# --- 메인 스레드 ---
if __name__ == "__main__":
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    t1 = threading.Thread(target=yolo_thread_func, args=(model,))
    t2 = threading.Thread(target=tcp_eog_thread_func)
    t3 = threading.Thread(target=tts_thread_func, daemon=True)

    t1.start(); t2.start(); t3.start()
    
    print("\n" + "="*40)
    print(" [시스템 시작] C# 연결 후 캘리브레이션을 진행하세요.")
    print(f" {Dwelling_Threshold}초 동안 머문 시선에 있는 물체 읽기")
    print(" - 'q' 키: 프로그램 종료")
    print("="*40 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            with frame_lock:
                latest_frame = frame.copy()

            # 1. YOLO 결과
            current_results = None
            with yolo_results_lock:
                if latest_yolo_results:
                    current_results = latest_yolo_results

            if current_results:
                for data in current_results.boxes.data.tolist():
                    conf = float(data[4])
                    if conf < CONFIDENCE_THRESHOLD: continue
                    
                    xmin, ymin, xmax, ymax = map(int, data[:4])
                    cls_id = int(data[5])
                    label = model.names[cls_id]
                    
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (xmin, ymin-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2)

            # 2. 시선(EOG) 그리기
            with eog_gaze_lock:
                gx, gy = latest_eog_gaze_data['x'], latest_eog_gaze_data['y']
            
            cv2.line(frame, (gx-10, gy), (gx+10, gy), RED, 2)
            cv2.line(frame, (gx, gy-10), (gx, gy+10), RED, 2)
            cv2.circle(frame, (gx, gy), 8, RED, 2)

            # 3. 상태 메시지 표시 안할 거임
            #if LAST_LABEL_MSG and (time.time() - LAST_LABEL_TIME) < LABEL_MSG_DURATION:
            #    cv2.putText(frame, LAST_LABEL_MSG, (20, 50) 
            #                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)

            cv2.imshow('User point of view', frame)

            # --- [수정된 부분] 3. Dwell Time 로직 (키 입력 제거, 자동 인식) ---
            
            # 현재 시선에 있는 물체 확인
            current_obj, _ = label_at_gaze(current_results, gx, gy, model.names)
            
            progress = 0.0 # 시각화용 진행률

            if current_obj:
                # 1. 보고 있던 물체를 계속 보는 경우
                if current_obj == dwell_target_name:
                    # 경과 시간 계산
                    elapsed = time.time() - dwell_time_start
                    
                    # 진행률 계산 (0.0 ~ 1.0)
                    progress = min(elapsed / Dwelling_Threshold, 1.0)

                    # 시간이 다 찼고 + 아직 이번 턴에 처리를 안 했다면?
                    if elapsed >= Dwelling_Threshold and not dwell_triggered:
                        
                        # [추가된 로직] 쿨다운 체크
                        # "지금 인식된 물체가 방금 말한 물체와 다르거나" 또는 "마지막으로 말한 지 3초가 지났으면" 실행
                        current_time = time.time()
                        if (current_obj != last_spoken["msg"]) or \
                           ((current_time - last_spoken["t"]) > TTS_COOLDOWN):
                            
                            print(f"👀 인식 완료 및 출력: {current_obj}")
                            tts_queue.put(current_obj)
                            
                            # 마지막으로 말한 정보 갱신
                            last_spoken["msg"] = current_obj
                            last_spoken["t"] = current_time
                        
                        else:
                            print(f"🔇 인식은 됐지만 최근에 말해서 생략함: {current_obj}")

                        # 소리를 냈든 안 냈든, 이 물체를 계속 보고 있는 상태에서는 다시 진입 금지
                        dwell_triggered = True
                
                # 2. 새로운 물체로 시선 이동
                else:
                    dwell_target_name = current_obj
                    dwell_time_start = time.time()
                    dwell_triggered = False
                    progress = 0.0
            
            else:
                # 3. 허공을 보는 경우 (초기화)
                dwell_target_name = None
                dwell_time_start = 0
                dwell_triggered = False
                progress = 0.0

            # 4. 시각화 (십자선 + Dwell 게이지)
            cv2.line(frame, (gx-10, gy), (gx+10, gy), RED, 2)
            cv2.line(frame, (gx, gy-10), (gx, gy+10), RED, 2)
            
            # Dwell 게이지 그리기 (노란색 링)
            draw_dwell_ui(frame, (gx, gy), progress, dwell_triggered)

            cv2.imshow('User point of view', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        is_running = False
        cap.release()
        cv2.destroyAllWindows()
        # 임시 파일 삭제 시도
        try:
            if os.path.exists("temp_voice.mp3"):
                os.remove("temp_voice.mp3")
        except: pass
        print("프로그램 종료.")