from ultralytics import YOLO
import cv2
import numpy as np

def main():
    # YOLOv8nモデルをロード（初回実行時にダウンロードされます）
    model = YOLO('yolov8n.pt')

    # Webカメラを開く
    cap = cv2.VideoCapture(0)

    try:
        while True:
            # フレームを読み込む
            ret, frame = cap.read()
            if not ret:
                break

            # 物体検出を実行
            results = model(frame)
            
            # 結果を描画
            annotated_frame = results[0].plot()
            
            # 結果を表示
            cv2.imshow('YOLOv8 Detection', annotated_frame)
            
            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # リソースを解放
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 