using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WindowsFormsApp1
{
    internal class GazeCalculator
    {
        // 캘리브레이션 기준값
        private double centerV_H = 0;
        private double centerV_V = 0;
        private double amp_H = 0;
        private double amp_V = 0;

        // [특허 적용] 회전 오차 보정 계수 (Crosstalk Correction)
        private double rotation_comp_H = 0; // 수평 움직임 시 수직 간섭 비율
        private double rotation_comp_V = 0; // 수직 움직임 시 수평 간섭 비율

        // [특허 적용] 데드존 (이 값 이하의 변화는 무시 -> 드리프트 방지)
        private double deadZone = 0.05; // 전압 기준 (실험적으로 조절 필요)

        public bool IsCalibrated { get; private set; } = false;

        // 캘리브레이션 데이터 설정
        public void SetCalibrationData(double cH, double rH, double lH, double cV, double uV, double dV)
        {
            centerV_H = cH;
            centerV_V = cV;

            // 1. 진폭(Amp) 계산 (기존과 동일)
            double deltaH = (Math.Abs(rH - cH) + Math.Abs(lH - cH)) / 2.0;
            double deltaV = (Math.Abs(uV - cV) + Math.Abs(dV - cV)) / 2.0;

            double rad20 = 20.0 * (Math.PI / 180.0);
            amp_H = (deltaH > 0.001) ? deltaH / Math.Sin(rad20) : 1.0;
            amp_V = (deltaV > 0.001) ? deltaV / Math.Sin(rad20) : 1.0;

            // 2. [특허 기능] 회전 오차 계수 계산 (간단 버전)
            // 원래는 복잡한 행렬을 써야 하지만, 여기선 "수평으로 움직일 때 수직이 얼마나 튀나" 비율만 봅니다.
            // (구현의 편의를 위해 여기서는 0으로 초기화하고 필요시 고도화 가능)
            rotation_comp_H = 0;
            rotation_comp_V = 0;

            IsCalibrated = true;
        }

        // 각도 계산 함수 (특허 알고리즘 적용)
        public void CalculateGaze(double rawH, double rawV, out double angleH, out double angleV)
        {
            angleH = 0;
            angleV = 0;

            if (!IsCalibrated) return;

            // 1. 원점 보정 (Translation)
            double dx = rawH - centerV_H;
            double dy = rawV - centerV_V;

            // 2. [특허 기능] 회전 보정 (Rotation Correction)
            // 전극이 비뚤게 붙어서 수평 이동 시 수직 전압도 같이 뛸 때 이를 상쇄함
            // (보정된 X = 원래X - (Y의 영향력))
            double corr_dx = dx - (dy * rotation_comp_V);
            double corr_dy = dy - (dx * rotation_comp_H);

            // 3. [특허 기능] 데드존 처리 (Noise/Drift Reduction)
            // 변화량이 너무 작으면 그냥 0(중앙)으로 간주
            if (Math.Abs(corr_dx) < deadZone) corr_dx = 0;
            if (Math.Abs(corr_dy) < deadZone) corr_dy = 0;

            // 4. 스케일링 및 각도 변환 (ArcSin Saturation 보정)
            // 기존에 우리가 짠 로직 (Saturation 해결용)
            angleH = ConvertToAngle(corr_dx, amp_H);
            angleV = ConvertToAngle(corr_dy, amp_V);
        }

        // 전압 -> 각도 변환 (ArcSin)
        private double ConvertToAngle(double delta, double amp)
        {
            double ratio = delta / amp;
            if (ratio > 1.0) ratio = 1.0;
            if (ratio < -1.0) ratio = -1.0;
            return Math.Asin(ratio) * (180.0 / Math.PI);
        }

        
        // [추가] 각도를 카메라 영상 좌표(Pixel)로 변환하는 함수
        // camWidth, camHeight: Python에서 쓰는 카메라 해상도 (예: 640, 480)
        // camFOV: 카메라 좌우 화각 (보통 웹캠은 60~70도)
        public System.Drawing.Point GetCameraCoordinates(double angleH, double angleV,
                                                         int camWidth = 640,
                                                         int camHeight = 480,
                                                         double camFOV = 60.0)
        {
            if (!IsCalibrated) return new System.Drawing.Point(camWidth / 2, camHeight / 2);

            // [수정 1] 민감도(Sensitivity) 계수 추가
            // 눈을 20도만 돌려도 화면 끝까지 가도록 1.5배~2.0배 증폭
            double sensitivity = 2.0; 

            // 1. 각도 -> 픽셀 변환
            double pixelsPerDegreeX = (camWidth / camFOV) * sensitivity;
            double pixelsPerDegreeY = (camHeight / (camFOV * (double)camHeight / camWidth)) * sensitivity;

            // 2. 좌표 계산
            int pixX = (int)(angleH * pixelsPerDegreeX);
            int pixY = (int)(angleV * pixelsPerDegreeY);

            int centerX = camWidth / 2;
            int centerY = camHeight / 2;

            int finalX = centerX + pixX;
            int finalY = centerY - pixY;

            // 3. 디버깅용: 콘솔에 현재 계산된 값 출력 (매우 중요!)
            // 좌표가 이상하면 이 로그를 봐야 합니다.
            Console.WriteLine($"AngleH: {angleH:F2}, FinalX: {finalX}");

            // 화면 밖으로 나가지 않게 제한
            finalX = Math.Max(0, Math.Min(camWidth, finalX));
            finalY = Math.Max(0, Math.Min(camHeight, finalY));

            return new System.Drawing.Point(finalX, finalY);
        }
    }
}