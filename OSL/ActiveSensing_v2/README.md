# OSL (Odor Source Holding)

중앙 `(0,0)`에 냄새 소스가 있고, 멀어질수록 농도가 감소하는 2D 환경에서  
**센서 관측만으로 소스 근처에 오래 머무르게** 학습하는 DQN 데모입니다.  
로봇은 **선속도 고정**, **각속도만 이산 선택**합니다.

- **v0 (`OdorHold-v0`)**: 좌/우 **2개 점 센서** `(cL, cR)`
- **v1 (`OdorHold-v1`)**: **단일 점 센서 + 4방향 스캔**(Front/Left/Back/Right)으로 액티브 센싱
- **v2 (`OdorHold-v2`)**: **RUN vs CAST(좌/우 캐스팅)** 행동을 학습(2-action) + CAST 결과로 헤딩 보정

특히, v2에서는 evaluation(eps==0) 버전의 plot을 추가했고 init, mid, final이 아닌, init, final, best.pt를 생성함
---

## 버전별 관측 방식

### v0
- 센서 2개(좌/우)에서 **점(point) 농도**를 샘플링
- 관측: `(cL, cR)`를 최근 `stack_n` 스텝 스택  
  예) `stack_n=4` → `[cL,cR] x 4 = 8차원`

### v1
- 센서 1개가 매 스텝 **4방향(F/L/B/R)**으로 전환하며 **점(point) 농도**를 샘플링  
- 관측(스캔 방향 위상을 알려주기 위해 포함):
  - per-step: `[c, sin(phi), cos(phi)]`
  - stacked: `3 * stack_n`  
    예) `stack_n=4` → 12차원
- 렌더(GIF)에서 **현재 센서가 바라보는 방향 레이(ray)**가 표시됩니다.

### v2 (추가)

- 정책이 매 스텝 **RUN(이동) vs CAST(정지+좌/우 캐스팅)**를 선택하는 2-action 구조
- CAST는 시작되면 LRLR(또는 LLRR) 4step을 자동으로 수행하며, 노이즈 평균화를 통해 좌/우 비교 신호를 얻습니다.
- CAST 4step 종료 후 delta = meanL - meanR의 부호에 따라 헤딩을 ±cast_turn만큼 보정합니다.
- 관측(방향 라벨 제거):
  - per-step: [c, mode] (mode: 0=RUN, 1=CAST)
  - stacked: 2 * stack_n
    예) stack_n=4 → 8차원

---

## 파일 구성

- `odor_env.py`  
  **v0** Gymnasium 환경 (등록 ID: `OdorHold-v0`)

- `odor_env_v1.py`  
  **v1** Gymnasium 환경 (등록 ID: `OdorHold-v1`)  
  단일 센서 + 4방향 스캔, 렌더에 센서 방향 표시

- `odor_env_v2.py`  
  **v1** Gymnasium 환경 (등록 ID: `OdorHold-v2`)  
  RUN vs CAST(좌/우 캐스팅) 2-action, CAST 종료 후 헤딩 보정(cast_turn)

- `train.py`  
  DQN 학습 스크립트. `runs/<run_name>/` 아래에 결과 저장:
  - `checkpoints/ init.pt, mid.pt, final.pt`
  - `media/init.gif, best.gif, final.gif`
  - `metrics.csv`, `config.json` (env_kwargs 포함)

- `eval.py`  
  run 폴더의 **init/mid/final 모델을 불러와** greedy(ε=0)로 rollout 후  
  **trajectory(궤적) plot**을 `media/traj_*.png`로 저장 (옵션: `traj_*.json`).

---

## 모델 구조(요약)

입력: 스택된 센서값  
출력: 각 행동(각속도 이산 선택)에 대한 Q값

```

(obs_dim) → FC(256) → ReLU → FC(256) → ReLU → FC(act_dim)

````

- v0 예: `obs_dim = 2 * stack_n`
- v1 예: `obs_dim = 3 * stack_n`
- v2 예: `obs_dim = 2 * stack_n`

---

## 실행

### 1) 학습 (v1이 기본인 경우)
```bash
python train.py
````

예시(에피소드/노이즈):

```bash
python train.py --total-episodes 600 --sensor-noise 0.01
python train.py --run-name osl_demo --total-episodes 600 --sensor-noise 0.01 --scan-penalty 0.01
```

> v0/v1 선택은 `train.py`의 `env_id` 기본값 및 register entry_point 설정에 따라 달라집니다.
> 현재 구성:
>
> * v0: `env_id="OdorHold-v0"`, `entry_point="odor_env:OdorHoldEnv"`
> * v1: `env_id="OdorHold-v1"`, `entry_point="odor_env_v1:OdorHoldEnv"`
> * v2: `env_id="OdorHold-v2"`, `entry_point="odor_env_v2:OdorHoldEnv"`

### 2) 평가 + 궤적 저장 (init/mid/final 모두)

```bash
python eval.py --run_dir runs/odor_dqn_YYYYMMDD_HHMMSS --episodes 50
```

궤적 raw 데이터도 저장:

```bash
python eval.py --run_dir runs/odor_dqn_YYYYMMDD_HHMMSS --episodes 50 --save_json
```

저장 결과:

* `runs/<run_name>/media/traj_init.png`
* `runs/<run_name>/media/traj_best.png`
* `runs/<run_name>/media/traj_final.png`
* (옵션) `traj_*.json`

---

## 메모

* 평가에서 greedy(ε=0)는 “랜덤 탐험 없이 항상 `argmax Q` 행동만 선택”을 뜻합니다.
* `sensor_noise=0.01` 등 env 파라미터는 `config.json`의 `env_kwargs`로 저장되어 평가에도 동일 적용됩니다.
* 센서 농도는 **직선 적분**이 아니라, 센서 위치에서의 **점 샘플(point sample)** 입니다.
