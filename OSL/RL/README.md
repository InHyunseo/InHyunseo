
### config.json
- 사용한 주요 설정/하이퍼파라미터 저장

### metrics.csv (에피소드 단위 기록)
컬럼:
- `episode`
- `endured_step` : 해당 에피소드에서 버틴 step 수
- `shaped_return` : (reward_scale 적용된) shaping reward 누적합
- `success` : `truncated & not terminated`이면 1
- `actor_loss`
- `critic_loss`
- `total_loss`

### plots/training_curves.png
- 상단: steps/episode + shaped_return(우측 y축) 및 moving average
- 하단: actor_loss / critic_loss 및 moving average

### media/*.gif
- `render_mode="rgb_array"`로 프레임을 모아서 GIF로 저장
- GIF 롤아웃은 **deterministic=True**로 실행(매 step argmax 행동)

### plots/policy_rollout_preview.png
- init/mid/final 롤아웃의 대표 프레임(첫 프레임) 3장 비교

---

## 8) 실행 방법(How to run)

### 설치(예시)
- Python 3.8+
- 필수:
  - `gymnasium`
  - `numpy`
  - `torch`
  - `matplotlib`
- GIF 저장:
  - 권장: `imageio`
  - (fallback) `Pillow`가 있으면 imageio 없이도 저장 시도

### 실행
```bash
python train_cartpole_ac.py
