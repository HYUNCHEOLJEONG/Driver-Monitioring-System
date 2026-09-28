import numpy as np

# MediaPipe Face Mesh 기준 눈 핵심 랜드마크 인덱스
# (우측 눈, 좌측 눈)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye_landmarks):
    """
    Eye Aspect Ratio (EAR) 계산 공식
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    p1 = np.array(eye_landmarks[0]) # 눈 왼쪽 끝
    p2 = np.array(eye_landmarks[1]) # 위쪽 눈꺼풀 1
    p3 = np.array(eye_landmarks[2]) # 위쪽 눈꺼풀 2
    p4 = np.array(eye_landmarks[3]) # 눈 오른쪽 끝
    p5 = np.array(eye_landmarks[4]) # 아래쪽 눈꺼풀 2
    p6 = np.array(eye_landmarks[5]) # 아래쪽 눈꺼풀 1

    # 세로 방향 거리 2개 계산
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)

    # 가로 방향 거리 계산
    horizontal = np.linalg.norm(p1 - p4)

    # EAR 계산
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6) # 분모 0 방지
    return ear

def get_eye_landmarks(landmarks, image_width, image_height):
    """
    Mediapipe 결과에서 양쪽 눈의 픽셀 좌표를 추출
    """
    right_eye = []
    left_eye = []

    for idx in RIGHT_EYE_INDICES:
        pt = landmarks[idx]
        right_eye.append((int(pt.x * image_width), int(pt.y * image_height)))

    for idx in LEFT_EYE_INDICES:
        pt = landmarks[idx]
        left_eye.append((int(pt.x * image_width), int(pt.y * image_height)))

    return right_eye, left_eye