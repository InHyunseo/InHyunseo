
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WindowsFormsApp1
{
    internal class EOGprocess
    {
        // --- 필터 상태 저장을 위한 변수 ---
        private double lpf_prev_out = 0;

        // HPF 변수
        private double hpf_prev_in = 0;
        private double hpf_prev_out = 0;

        private double _gain = 1.0;

        // --- 필터 계수 (Alpha) ---
        // (12kHz 샘플링 속도 기준 계산 결과)

        // 1. 40Hz LPF (Cutoff Frequency = 40Hz)
        // 40Hz LPF for 12kHz Sampling:  0.0103;
        // 40Hz LPF for 1920Hz Sampling: 0.115;
        //private double alpha_lpf = 0.115;
        private double alpha_lpf = 0.0103;

        // 2. 0.5Hz HPF (Cutoff Frequency = 0.5Hz) 1.5HZ: 
        // 0.5Hz HPF for 12kHz Sampling:  0.999738;
        // 0.5Hz HPF for 1920Hz Sampling : 0.9983;
        //private double alpha_hpf = 0.9983;
        private double alpha_hpf = 0.999738;

        public void SetGain(double gain)
        {
            this._gain = gain;
        }
        /*
        public void SetHPF_Alpha(double alpha)
        {
            this.alpha_hpf = alpha;
        }
        */
        /// <summary>
        /// 새 원본(Raw) 데이터 1개를 받아 필터링된 값을 반환합니다.
        /// </summary>
        /// <param name="newSample">MSP430에서 받은 원본 ADC 값</param>
        /// <returns>필터링이 완료된 EOG 신호 값</returns
        /// 
        /// >
        public double ProcessSample(double newSample)
        {
            // 1단계: LPF (EMA) - 노이즈 제거
            double lpf_out = (alpha_lpf * newSample) + (1.0 - alpha_lpf) * lpf_prev_out;
            lpf_prev_out = lpf_out; // 다음 계산을 위해 현재 값을 저장

            // 2단계: HPF (1st Order IIR) - DC Drift 제거
            double hpf_out = alpha_hpf * (hpf_prev_out + lpf_out - hpf_prev_in);
            hpf_prev_in = lpf_out;    // 다음 계산을 위해 현재 입력 값 저장
            hpf_prev_out = hpf_out;   // 다음 계산을 위해 현재 출력 값 저장

            // 최종적으로 LPF, HPF를 모두 통과한 값을 반환
            return hpf_out*_gain;
        }
    }
}


/*
using System;
using System.Collections.Generic;
using System.Linq;

namespace WindowsFormsApp1
{
    internal class EOGprocess
    {
        private double prevOutput = 0;
        private bool isFirst = true;

        // === [튜닝 파라미터] ===
        // 1. 노이즈 기준값: 변화량이 이 값보다 작으면 '노이즈'로 간주하고 뭉갭니다.
        // 노이즈가 여전히 심하면 이 값을 30 -> 50 -> 80으로 올리세요.
        private double noiseThreshold = 50.0;

        // 2. 정지 상태일 때 필터 강도 (작을수록 부드러움, 0.01 ~ 0.1)
        private double alphaSlow = 0.05;

        // 3. 움직일 때 필터 강도 (클수록 빠름, 0.5 ~ 0.9)
        private double alphaFast = 0.8;

        public double ProcessSample(double newSample)
        {
            if (isFirst)
            {
                prevOutput = newSample;
                isFirst = false;
                return newSample;
            }

            // 1. 변화량 계산 (절대값)
            double diff = Math.Abs(newSample - prevOutput);

            // 2. 상황에 따른 Alpha(반영률) 결정 [핵심 로직]
            double currentAlpha;

            if (diff < noiseThreshold)
            {
                // 변화가 작다 = 노이즈다 -> 필터를 아주 세게 건다 (부드럽게)
                currentAlpha = alphaSlow;
            }
            else
            {
                // 변화가 크다 = 눈 움직임이다 -> 필터를 푼다 (빠르게)
                currentAlpha = alphaFast;
            }

            // 3. 지수 이동 평균 (EMA) 계산
            double output = (newSample * currentAlpha) + (prevOutput * (1.0 - currentAlpha));

            prevOutput = output;
            return output;
        }

        public void Reset()
        {
            isFirst = true;
            prevOutput = 0;
        }
    }
}
*/