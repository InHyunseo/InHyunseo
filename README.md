
---
## Ongoing Projects
### 🚗 Autonomous Systems & Robotics
| Period | Project | Description & Tech |
| :--- | :--- | :--- |
| 2026 상반기 · 진행 중 | E2E Autonomous Driving AI <br> _(Python, C++, ROS2, PyTorch, TensorRT, Jetson Orin Nano)_ | • Lane segmentation + front vehicle bbox 입력 기반 E2E 차선 주행 및 정지 차량 회피·추월 모델 <br> • SegFormer·YOLO fine-tune 후 freeze, ResNet18×2 + ControlHead/WaypointHead 멀티태스크 학습 <br> • rosbag 데이터 자동 라벨링, ONNX→TensorRT 변환 및 Jetson 실차 배포 파이프라인 |
| 2026 상반기 · 진행 중 | Inner Path Safety Verification Framework <br> _(Python, MATLAB/Simulink, PyTorch, MoE, SNN)_ | • 조건부 활성화 자율주행 AI의 내부 경로 관측 기반 안전 검증 프레임워크 <br> • APDA/APDM/APST/APIM 및 TAR index 지표 설계 <br> • ISO/PAS 8800·SOTIF 대응 SIL↔MIL closed-loop 검증 구조 |

### 💻 Programming & Community
| Period | Project | Description & Tech |
| :--- | :--- | :--- |
| 2026 상반기 · 진행 중 | Algorithm Problem Solving Open Chat <br> _(C++, Python)_ | • C++ / Python 기반 LeetCode-style algorithm problem solving 오픈채팅방 운영 <br> • 본인은 C++ 트랙 중심으로 coding interview practice 및 문제 풀이 공유 <br> • 자료구조, 알고리즘, 구현 중심의 꾸준한 problem-solving 스터디 운영 |

## Previous Projects
### 🚗 Autonomous Systems & Robotics
| Period | Project | Description & Tech |
| :--- | :--- | :--- |
| 2025 상반기 | Grid-based Automotive Algorithms <br> _(Simulink)_ | • A* path planning and a Pure Pursuit/PID control-based system |
| 2025 하반기 | 2025 Mobility Challenge: CAV Cooperative Driving <br> _(C++, Python, ROS2)_ | • 협력 자율주행(CAV) 교차로/합류 시나리오 알고리즘 설계 <br> • Conflict Zone, ETA, V2V Domain Bridge 기반 우선순위·양보 로직 구현 <br> • 동적 차선 계획, HV 대응 및 active safety 제어 |

### 🧠 AI Research & Bio-inspired Learning
| Period | Project | Description & Tech |
| :--- | :--- | :--- |
| 2026 상반기 | Scene-Conditional Dynamic Mask Networks — CIFAR Validation <br> _(PyTorch, ResNet18, CIFAR-10-C)_ | • Scene-conditioned channel mask로 context별 sub-network를 구성하는 SCDMN 구조 검증 <br> • gate-depth pattern, linear probe, mask-IoU 분석 파이프라인 구현 <br> • CIFAR regression negative result를 통해 실제 driving 데이터 검증 필요성 도출 |
| 2026 상반기 | Scene-Conditional Dynamic Mask Networks — Driving-scale Validation <br> _(PyTorch, ResNet18, comma2k19)_ | • comma2k19 주행 영상과 CAN steering angle 기반 driving-scale SCDMN 검증 <br> • day_clear / day_overcast / night context 자동 라벨링 및 steering regression 실험 <br> • 단일 모델 대비 context mask가 hard context의 mode averaging을 줄이는지 평가 |
| 2026 상반기 | Odor Source Localization (Larva Connectome RL) <br> _(Python, PyTorch, Gymnasium, PPO/SAC, Connectome)_ | • 양측 냄새 센서와 독립 head/body 회전축을 갖는 odor source localization RL 환경 구축 <br> • PPO/SAC curriculum, GRU·Drosophila larva connectome policy 실험 <br> • active sensing 행동의 dynamics, linear probe, causal ablation 분석 |
| 2025 상하반기 | Connectome-based AI for Drosophila <br> _(PyTorch, DoOR, Connectome)_ | • 초파리 후각 시스템 커넥톰 모방, 냄새 분류 AI 모델 <br> • AI model tuning and testing, from MLPs to RNNs to reservoir networks, validated with random models <br> • Exploring how brain networks can be used as pretrained ANN models |

### 🧠 Bioengineering & Embedded Systems
| Period | Project | Description & Tech |
| :--- | :--- | :--- |
| 2025 하반기 | Smart Goggles <br> _(MSP430, YOLO)_ | • EOG-based pointer + object detection AI <br> • PCB design for the EOG AFE (Analog Front End) circuit <br> • Used YOLOv8n and gTTS, with logic to handle EOG drift and blink artifacts |
| 2025 하반기 | AI Home Training <br> _(Jetson Orin, BlazePose)_ | • 실시간 관절점 인식 기반 자세 교정 디바이스 <br> • Jetson Orin Nano + TensorRT 가속 활용, 다인용 고성능 처리 <br> • 다인용 관절 각도 계산 알고리즘 및 시각/청각 피드백 시스템 |
| 2025 상반기 | Neural Stimulation <br> _(Research)_ | • 정전류 자극기 및 DC-DC converter 아날로그 회로 설계 및 in vivo 실험 <br> • 자극 안전성(Safety), Focality, Steerability 분석 및 시뮬레이션 |
| 2025 상반기 | Embedded & Analog Circuits <br> _(ATmega128, LTspice, PCB)_ | • ATmega128 기반 미니카 제작 (거리 센서 연동 모터/LED 제어) <br> • PPG/스피커 아날로그 회로 설계 및 ECG 측정 PCB 납땜/검증 <br> • LTspice 시뮬레이션 및 오실로스코프 등 전자계측 장비 활용 |
---

## 🔬 Research Interests
* Brain-inspired AI system: 뇌모방 인공지능
* AI model interpretability & verification: AI 모델 내부 분석 및 검증
* E2E AI Automotive system: 자율주행, 자율지능 시스템
* System E2E design: 통합 시스템 프로토타입 개발

## 🛠️ Technical Skills
| Area | Skills |
| :--- | :--- |
| Programming | Python, C++, MATLAB |
| AI & Robotics | PyTorch, ROS2, TensorRT |
| Modeling & Simulation | Simulink, LTspice, JMAG, AutoCAD, Gazebo |
| Embedded Boards | Jetson Orin NX, Jetson Orin Nano, Raspberry Pi 5 |
| Microcontrollers | MSP430, ATmega128 |
| Development Environment | Ubuntu/Linux, Git, Docker |
| Collaboration & Documentation | Notion, Slack |
| Hardware | PCB design, soldering, oscilloscope, function generator, power supply, digital multimeter |
