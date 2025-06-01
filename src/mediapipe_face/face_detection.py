import cv2
import mediapipe as mp

def main():
    # MediaPipeの顔検出を初期化
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # Webカメラを開く
    cap = cv2.VideoCapture(0)
    
    # 顔検出モデルを初期化
    with mp_face_detection.FaceDetection(
        model_selection=0,  # 0=近距離モデル, 1=遠距離モデル
        min_detection_confidence=0.5  # 検出の信頼度閾値
    ) as face_detection:
        
        while True:
            # フレームを読み込む
            ret, frame = cap.read()
            if not ret:
                break
                
            # BGR→RGBに変換（MediaPipeはRGBを使用）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 顔検出を実行
            results = face_detection.process(frame_rgb)
            
            # 検出結果を描画
            if results.detections:
                for detection in results.detections:
                    # 顔の周りに枠と特徴点を描画
                    mp_drawing.draw_detection(frame, detection)
                    
                    # 検出の信頼度を表示
                    confidence = detection.score[0]
                    position = detection.location_data.relative_bounding_box
                    height, width, _ = frame.shape
                    x = int(position.xmin * width)
                    y = int(position.ymin * height)
                    
                    # 信頼度をテキストとして表示
                    cv2.putText(frame, f'Confidence: {confidence:.2f}',
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (0, 255, 0), 2)
            
            # 結果を表示
            cv2.imshow('MediaPipe Face Detection', frame)
            
            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 