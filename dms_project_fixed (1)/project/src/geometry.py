"""좌표 변환 공용 유틸. 눈 개폐 등은 반드시 픽셀 좌표로 변환 후 계산한다."""
import numpy as np


def landmarks_to_pixels(landmarks, width, height):
    """정규화 랜드마크 리스트 -> (N, 2) float 픽셀 좌표 배열 (종횡비 왜곡 방지)."""
    return np.array([[p.x * width, p.y * height] for p in landmarks], dtype=np.float32)
