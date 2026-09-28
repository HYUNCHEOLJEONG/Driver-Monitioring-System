import cv2
import mediapipe as mp
import numpy as np
from src.utils import get_eye_landmarks, calculate_ear

def main():
    # MediaPipe 초기화 (안정적인 solutions API 활용)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 웹캠 연결 (0번 카메라)
    cap = cv2.VideoCapture(0)
    
    # 임계값 설정 (실제 환경에 따라 미세 조정 필요)
    EAR_THRESHOLD = 0.22      # 이 수치 이하로 떨어지면 눈이 감긴 것임
    DROWSY_FRAMES_LIMIT = 30  # 감긴 상태가 지속되는 프레임 수 (약 1초~1.5초)
    drowsy_counter = 0

    print("[INFO] DMS Phase 1 (EAR 졸음 감지) 시스템 가동 시작...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("웹캠 프레임을 받아올 수 없습니다.")
            break

        # 좌우 반전 (거울 모드) 및 RGB 변환
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 모델 추론
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 눈 랜드마크 추출
                right_eye, left_eye = get_eye_landmarks(face_landmarks.landmark, w, h)

                # 양쪽 눈 EAR 계산
                right_ear = calculate_ear(right_eye)
                left_ear = calculate_ear(left_eye)
                avg_ear = (right_ear + left_ear) / 2.0

                # 시각적으로 눈 위치에 초록색 라인 그려주기
                cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)

                # 졸음 판단 로직
                if avg_ear < EAR_THRESHOLD:
                    drowsy_counter += 1
                    if drowsy_counter >= DROWSY_FRAMES_LIMIT:
                        cv2.putText(frame, "WARNING: DROWSINESS DETECTED!", (30, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                else:
                    drowsy_counter = 0

                # 화면에 실시간 EAR 수치 출력
                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 결과 화면 출력
        cv2.imshow("DMS Phase 1 - Drowsiness Detection", frame)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()