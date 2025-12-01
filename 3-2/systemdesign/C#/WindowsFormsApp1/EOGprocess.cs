using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WindowsFormsApp1
{
    internal class EOGprocess
    {       // --- 필터 상태 저장을 위한 변수 ---
            // (필터는 '직전 값'을 기억해야 합니다)

            // LPF (Low-Pass Filter) 변수 (근육/전원 노이즈 제거용)
            private double lpf_prev_out = 0;

            // HPF (High-Pass Filter) 변수 (DC Drift 제거용 - EOG에 필수!)
            private double hpf_prev_in = 0;
            private double hpf_prev_out = 0;

            // --- 필터 계수 (Alpha) ---
            // (이 값은 1kHz 샘플링 속도 기준으로 계산되었습니다)
            // (샘플링 속도가 1kHz (TACCR0=6000)가 아니면 이 계수들은 다시 계산해야 합니다!)

            // 1. 40Hz LPF (Cutoff Frequency = 40Hz)
            // T = 1/1000s, T_c = 1/(2*pi*40) = 0.00397s
            // alpha_lpf = T / (T_c + T) = 0.001 / (0.00397 + 0.001) = 0.201
            private double alpha_lpf = 0.201;

            // 2. 0.5Hz HPF (Cutoff Frequency = 0.5Hz)
            // T = 1/1000s, T_c = 1/(2*pi*0.5) = 0.318s
            // alpha_hpf = T_c / (T_c + T) = 0.318 / (0.318 + 0.001) = 0.9968
            private double alpha_hpf = 0.9968;


            /// <summary>
            /// 새 원본(Raw) 데이터 1개를 받아 필터링된 값을 반환합니다.
            /// </summary>
            /// <param name="newSample">MSP430에서 받은 원본 ADC 값</param>
            /// <returns>필터링이 완료된 EOG 신호 값</returns>
            public double ProcessSample(double newSample)
            {
                // 1단계: LPF (EMA) - 노이즈 제거
                // y[n] = (alpha * x[n]) + (1 - alpha) * y[n-1]
                double lpf_out = (alpha_lpf * newSample) + (1.0 - alpha_lpf) * lpf_prev_out;
                lpf_prev_out = lpf_out; // 다음 계산을 위해 현재 값을 저장

                // 2단계: HPF (1st Order IIR) - DC Drift 제거
                // y[n] = alpha * (y[n-1] + x_lpf[n] - x_lpf[n-1])
                double hpf_out = alpha_hpf * (hpf_prev_out + lpf_out - hpf_prev_in);
                hpf_prev_in = lpf_out;    // 다음 계산을 위해 현재 입력 값 저장
                hpf_prev_out = hpf_out;   // 다음 계산을 위해 현재 출력 값 저장

                // 최종적으로 LPF, HPF를 모두 통과한 값을 반환
                return hpf_out;
            }
        }
}
