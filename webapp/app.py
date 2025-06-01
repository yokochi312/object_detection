import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from PIL import Image
import tempfile
import os
from datetime import datetime

# セッション状態の初期化
if 'snapshot_counter' not in st.session_state:
    st.session_state.snapshot_counter = 0
if 'last_snapshot' not in st.session_state:
    st.session_state.last_snapshot = None
if 'camera_frame' not in st.session_state:
    st.session_state.camera_frame = None

# 保存用ディレクトリの作成
if not os.path.exists('captured_images'):
    os.makedirs('captured_images')

# ページの設定
st.set_page_config(
    page_title="AIによる物体・顔検出アプリ",
    page_icon="🔍",
    layout="wide"
)

# カスタムCSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .detection-result {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# タイトルと説明
st.title("🔍 AIによる物体・顔検出アプリ")
st.markdown("""
    このアプリでは、画像やカメラからリアルタイムで物体検出と顔検出を行うことができます。
    - **物体検出**: YOLOv8を使用して80種類以上の物体を検出
    - **顔検出**: MediaPipeを使用して高精度な顔検出を実行
""")

# サイドバーの設定
with st.sidebar:
    st.header("⚙️ 設定")
    detection_mode = st.selectbox(
        "検出モードを選択",
        ["物体検出 (YOLOv8)", "顔検出 (MediaPipe)"],
        help="使用する検出モードを選択してください"
    )
    
    st.markdown("---")
    st.markdown("### 📝 モード説明")
    if detection_mode == "物体検出 (YOLOv8)":
        st.info("""
            **YOLOv8による物体検出**
            - 人、車、動物などの80種類以上の物体を検出
            - 各物体の位置と確率を表示
            - リアルタイムでの検出が可能
        """)
    else:
        st.info("""
            **MediaPipeによる顔検出**
            - 高精度な顔検出
            - 複数の顔を同時に検出
            - 軽量で高速な処理
        """)

# MediaPipeの初期化
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

def process_image_yolo(image, model):
    results = model(image)
    return results[0].plot()

def process_image_mediapipe(image):
    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as face_detection:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        
        output_image = image.copy()
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(output_image, detection)
                
                # 検出確率を表示
                score = detection.score[0]
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = output_image.shape
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                cv2.putText(output_image, f'Confidence: {score:.2f}', 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 0), 2)
        
        return output_image

def save_image(image, detection_type):
    """画像を保存する関数"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_images/{detection_type}_{timestamp}.jpg"
    cv2.imwrite(filename, image)
    return filename

# メインコンテンツ
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📸 入力")
    source = st.radio("入力ソースを選択", ["カメラ", "画像をアップロード"])
    
    if source == "画像をアップロード":
        uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'jpeg', 'png'], key="uploader")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.image(uploaded_file, caption="アップロードされた画像", use_column_width=True)
            
            with st.spinner("画像を処理中..."):
                if detection_mode == "物体検出 (YOLOv8)":
                    model = load_yolo_model()
                    result_image = process_image_yolo(image, model)
                else:
                    result_image = process_image_mediapipe(image)
                
                with col2:
                    st.markdown("### 🎯 検出結果")
                    st.image(result_image, caption="検出結果", use_column_width=True)
                    
                    # アップロードされた画像の検出結果をダウンロード
                    st.download_button(
                        label="📥 検出結果をダウンロード",
                        data=cv2.imencode('.jpg', result_image)[1].tobytes(),
                        file_name=f"detection_result.jpg",
                        mime="image/jpeg",
                        key="download_upload_result"
                    )
    
    else:  # カメラ使用
        st.warning("⚠️ カメラを使用する場合は、カメラへのアクセスを許可してください")
        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            run = st.checkbox("カメラを起動", key="camera_running")
        with col_cam2:
            if st.button("📸 撮影", disabled=not run, key="snapshot_button"):
                st.session_state.snapshot_counter += 1
        
        camera_placeholder = st.empty()
        
        if run:
            cap = cv2.VideoCapture(0)
            if detection_mode == "物体検出 (YOLOv8)":
                model = load_yolo_model()
            
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("カメラからの映像の取得に失敗しました")
                    break
                
                # 検出処理
                if detection_mode == "物体検出 (YOLOv8)":
                    processed_frame = process_image_yolo(frame, model)
                else:
                    processed_frame = process_image_mediapipe(frame)
                
                # スナップショット撮影
                if st.session_state.snapshot_counter > 0:
                    st.session_state.last_snapshot = processed_frame.copy()
                    detection_type = "yolo" if detection_mode == "物体検出 (YOLOv8)" else "face"
                    saved_path = save_image(processed_frame, detection_type)
                    st.success(f"画像を保存しました！: {saved_path}")
                    st.session_state.snapshot_counter = 0
                    
                    # 撮影した画像を結果カラムに表示
                    with col2:
                        st.markdown("### 📸 撮影結果")
                        st.image(st.session_state.last_snapshot, caption="撮影された画像", use_column_width=True)
                        st.download_button(
                            label="📥 撮影画像をダウンロード",
                            data=cv2.imencode('.jpg', st.session_state.last_snapshot)[1].tobytes(),
                            file_name="snapshot.jpg",
                            mime="image/jpeg",
                            key="download_snapshot_result"
                        )
                
                camera_placeholder.image(processed_frame, channels="BGR")
            
            cap.release()

# フッター
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ using Streamlit, YOLOv8, and MediaPipe</p>
    </div>
""", unsafe_allow_html=True) 