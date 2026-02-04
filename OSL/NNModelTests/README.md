````md
# NNModelTests

`OdorHold-v3` 환경에서 **신경망 구조/메모리 방식**을 바꿔가며 성능을 비교하는 실험 폴더입니다.  
비교 축은 “과거 정보(메모리)를 어떻게 모델에 제공하느냐”입니다.

---

## 실험 분류 (총 3가지)

### 1) (기존) **MLP + 4 stack memory**
- **구성**: MLP(QNet) + `stack_n=4`
- **메모리 의미**: 최근 4스텝의 관측을 **프레임 스택**으로 붙여서 입력에 제공  
  → MLP는 자체 기억이 없으므로 “과거 정보”를 입력 차원으로 강제로 넣는 방식
- **장점**
  - 구현이 가장 단순 (표준 DQN 형태)
  - CAST/RUN 직후 변화 같은 단기 패턴을 비교적 잘 잡음
- **단점**
  - `stack_n`을 늘리면 입력 차원이 커짐(학습 샘플 효율/일반화 저하 가능)
  - 스택 길이보다 긴 의존성은 표현 불가

---

### 2) **MLP + single stack memory**
- **구성**: MLP(QNet) + `stack_n=1`
- **메모리 의미**: 사실상 **메모리 없음**(현재 스텝 관측만 사용)
- **장점**
  - 입력이 작고 빠름, 베이스라인으로 명확
  - “현재 관측만으로 가능한지”를 확인하는 실험에 적합
- **단점**
  - POMDP 성격이 강하면(특히 CAST 패턴/직전 정보 필요) 학습이 불안정하거나 성능 한계가 큼
  - CAST 직후 TURN 결정처럼 “직전 스캔 결과/상태”가 필요하면 취약

---

### 3) **RNN(GRU) + single stack**
- **구성**: GRU(RQNet) + `stack_n=1` + (학습 시) `seq_len` 시퀀스 샘플링(DRQN 스타일)
- **메모리 의미**: 과거 정보는 입력 스택이 아니라 **hidden state**로 내부에 저장
- **장점**
  - 입력은 작게 유지하면서도 과거 의존성 학습 가능
  - CAST 과정/직후 결정 같은 시계열 구조를 자연스럽게 다룸
- **단점**
  - 구현/디버깅 난이도 증가(episode replay, seq_len, hidden reset 등)
  - 하이퍼파라미터(`seq_len`, `batch_size`, `hidden`) 영향이 큼

---

## 실행 요약

- **1번**: `train.py` (또는 `train_MLP.py`) + `--stack-n 4`
- **2번**: `train.py` + `--stack-n 1`
- **3번**: `train_rnn.py` + `--stack-n 1` + `--seq-len <T>`

평가/시각화는 공통으로:
```bash
python eval.py --run_dir runs/<RUN_NAME> --episodes 30 --seed_base 100000
````

---

## 출력물

각 run은 `runs/<RUN_NAME>/`에 저장됩니다.

* `config.json`, `metrics.csv`, `metrics_eval.csv`
* `checkpoints/ (init/best/final)`
* `plots/`, `media/` (gif, trajectory png 등)

```
```
