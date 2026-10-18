"""좌표 변환 공용 유틸. 눈 개폐 등은 반드시 픽셀 좌표로 변환 후 계산한다."""
import numpy as np


def landmarks_to_pixels(landmarks, width, height):
    """정규화 랜드마크 리스트 -> (N, 2) float 픽셀 좌표 배열 (종횡비 왜곡 방지)."""
    return np.array([[p.x * width, p.y * height] for p in landmarks], dtype=np.float32)


def eye_aspect_ratio(p):
    """EAR. p: (6,2) 픽셀 좌표, 순서 [외측 안각, 위1, 위2, 내측 안각, 아래2, 아래1].
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return float((v1 + v2) / (2.0 * h + 1e-6))


def calculate_head_pose_angles(pixels):
    """
    픽셀 좌표계에서 코(1)와 양쪽 눈(33, 263)의 위치를 이용해 
    고개의 좌우(Yaw) 및 상하(Pitch) 회전 편차를 계산합니다.
    """
    if pixels is None or len(pixels) <= 263:
        return 0.0, 0.0

    nose = pixels[1]
    left_eye = pixels[33]
    right_eye = pixels[263]

    eye_center = (left_eye + right_eye) / 2.0
    diff = nose - eye_center

    yaw = float(diff[0])
    pitch = float(diff[1])
    
    return yaw, pitch