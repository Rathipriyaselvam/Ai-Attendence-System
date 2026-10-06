import streamlit as st
from config import config, save_settings
from src.face_recognition import get_registered_embeddings

def render_settings_page():
    st.markdown("""
        <div class="main-header">
            <h1>⚙️ System Configuration & Model Settings</h1>
            <p>Customize AI Recognition Models, Detector Backends, Distance Thresholds, and System Parameters.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.form("settings_form"):
        st.subheader("1. AI Face Recognition Model Settings")

        col1, col2 = st.columns(2)

        models = ["Facenet512", "Facenet", "VGG-Face", "ArcFace", "OpenFace"]
        detectors = ["opencv", "ssd", "mtcnn", "retinaface"]
        metrics = ["cosine", "euclidean", "euclidean_l2"]

        cur_model = config.get("model_name", "Facenet512")
        cur_detector = config.get("detector_backend", "opencv")
        cur_metric = config.get("distance_metric", "cosine")

        with col1:
            model_name = st.selectbox(
                "Recognition Model",
                models,
                index=models.index(cur_model) if cur_model in models else 0,
                help="DeepFace representation backbone. 'Facenet512' provides high accuracy and fast speed."
            )
            detector_backend = st.selectbox(
                "Face Detector Backend",
                detectors,
                index=detectors.index(cur_detector) if cur_detector in detectors else 0,
                help="'opencv' is lightweight and fast for webcams."
            )

        with col2:
            distance_metric = st.selectbox(
                "Distance Metric",
                metrics,
                index=metrics.index(cur_metric) if cur_metric in metrics else 0,
                help="'cosine' distance is recommended for Facenet embedding vectors."
            )
            rec_threshold = st.slider(
                "Recognition Distance Threshold",
                min_value=0.20,
                max_value=0.80,
                value=float(config.get("recognition_threshold", 0.45)),
                step=0.05,
                help="Threshold for vector distance comparison. Lower = Stricter. Higher = More lenient."
            )

        st.markdown("---")
        st.subheader("2. Performance & Webcam Settings")

        p_col1, p_col2 = st.columns(2)

        with p_col1:
            cooldown_sec = st.number_input(
                "Attendance Cooldown Period (seconds)",
                min_value=5,
                max_value=3600,
                value=int(config.get("attendance_cooldown_seconds", 60)),
                step=5,
                help="Time window to prevent duplicate logging for the same student."
            )
            frame_skip = st.select_slider(
                "Process Every N Frames (Frame Skipping)",
                options=[1, 2, 3, 5, 10],
                value=int(config.get("process_every_n_frames", 3)),
                help="Higher value skips face embedding computation on N-1 intermediate frames for higher webcam FPS."
            )

        with p_col2:
            camera_index = st.number_input(
                "Webcam Camera Device Index",
                min_value=0,
                max_value=5,
                value=int(config.get("camera_index", 0)),
                step=1,
                help="0 is usually default integrated webcam. 1 or 2 for external USB webcams."
            )
            frame_width = st.selectbox(
                "Processing Frame Width (Pixels)",
                [480, 640, 800, 1280],
                index=[480, 640, 800, 1280].index(config.get("frame_width", 640)),
                help="Resizing frames before face detection accelerates processing speed."
            )

        st.markdown("---")

        submit = st.form_submit_button("💾 Save & Apply Settings", type="primary", use_container_width=True)

        if submit:
            new_config = {
                "model_name": model_name,
                "detector_backend": detector_backend,
                "distance_metric": distance_metric,
                "recognition_threshold": rec_threshold,
                "attendance_cooldown_seconds": cooldown_sec,
                "process_every_n_frames": frame_skip,
                "camera_index": camera_index,
                "frame_width": frame_width
            }

            if save_settings(new_config):
                config.update(new_config)
                # Force reload embeddings cache for new model if changed
                get_registered_embeddings(model_name=model_name, force_reload=True)
                st.success("✅ Configuration settings saved and applied successfully!")
            else:
                st.error("Failed to save settings file.")
