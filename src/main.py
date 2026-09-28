import cv2
import mediapipe as mp

def main():
    # 1. MediaPipe Face Mesh 초기화 (얼굴 랜드마크 정밀 추출 도구 준비)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,           # 화면에 감지할 최대 얼굴 수 (운전자 1명이므로 1)
        refine_landmarks=True,     # 눈동자 주변까지 정밀하게 감지
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 2. 랜드마크를 시각화(그려주는) 도구 준비
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # 3. 웹캠 연결 (0번은 기본 노트북/외장 웹캠)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다. 카메라 연결을 확인해주세요.")
        return

    print("=== DMS 1~2주차 파이프라인 테스트 구동 중 (종료하려면 키보드의 'q'를 누르세요) ===")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("프레임을 읽어오지 못했습니다.")
            break

        # 성능 최적화를 위해 프레임 쓰기 금지 설정
        frame.flags.writeable = False
        # OpenCV의 기본 BGR 색상을 MediaPipe가 요구하는 RGB 색상으로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # AI 모델 추론 수행 (얼굴 랜드마크 좌표 계산)
        results = face_mesh.process(rgb_frame)

        # 화면에 그림을 그리기 위해 다시 쓰기 허용 및 BGR 색상으로 복원
        frame.flags.writeable = True
        output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # 얼굴 랜드마크가 감지되었다면 화면에 그물망(그리기) 출력
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

        # 결과 화면을 창으로 띄우기
        cv2.imshow('DMS Pipeline Test - Team Jeonbang', output_frame)

        # 키보드 'q'를 누르면 창이 닫히며 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()