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


# ── 머리 자세 (solvePnP) ─────────────────────────────────────────
# 일반적인 얼굴 3D 모델(mm, 카메라 좌표계: x 오른쪽, y 아래, z 카메라에서 멀어짐). 정면 = 항등 회전.
_MODEL_3D = np.array([
    [0.0,    0.0,   0.0],    # 코끝            (MediaPipe 1)
    [0.0,  330.0,  65.0],    # 턱 끝           (152)
    [-225.0, -170.0, 135.0], # 화면 왼쪽 눈 바깥 (33)
    [225.0,  -170.0, 135.0], # 화면 오른쪽 눈 바깥(263)
    [-150.0, 150.0, 125.0],  # 화면 왼쪽 입꼬리 (61)
    [150.0,  150.0, 125.0],  # 화면 오른쪽 입꼬리(291)
], dtype=np.float64)
_POSE_IDX = [1, 152, 33, 263, 61, 291]


def calculate_head_pose_angles(pts, width=None, height=None):
    """픽셀 랜드마크 (N,2) -> (yaw, pitch) 도 단위.

    yaw   : +  = 사용자가 자기 왼쪽으로 고개를 돌림 (코가 화면 오른쪽으로 이동)
    pitch : +  = 고개 숙임 (코가 화면 아래로 이동)
    gaze.py 의 부호 규칙(yaw>thr=LEFT, yaw<-thr=RIGHT, pitch>thr=DOWN)과 같다.
    width/height 를 주면 카메라 내부 파라미터를 정확히 근사한다(생략 시 랜드마크 범위로 추정).
    """
    import cv2
    img = np.asarray(pts, dtype=np.float64)[_POSE_IDX]
    if width is None or height is None:
        width, height = max(float(pts[:, 0].max()), 1.0) * 1.2, max(float(pts[:, 1].max()), 1.0) * 1.2
    f = float(width)
    cam = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, img, cam, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    fwd = R @ np.array([0.0, 0.0, -1.0])      # 얼굴이 향하는 방향(카메라 쪽)
    yaw = float(np.degrees(np.arctan2(fwd[0], -fwd[2])))
    pitch = float(np.degrees(np.arctan2(fwd[1], -fwd[2])))
    return yaw, pitch