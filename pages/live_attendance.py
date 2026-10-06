import streamlit as st
import cv2
import numpy as np
import time
import datetime
from PIL import Image

from config import config
from src.face_detection import detect_faces_opencv, draw_recognition_box, resize_frame
from src.face_recognition import (
    extract_face_embedding,
    match_face_embedding,
    get_registered_embeddings
)
from src.attendance import mark_live_attendance
from src.database import get_student

def render_live_attendance_page():
    st.markdown("""
        <div class="main-header">
            <h1>📷 Live Webcam Attendance</h1>
            <p>Real-Time Face Detection, Recognition & Automatic Logging</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls & Parameters
    st.sidebar.markdown("### ⚙️ Live Controls")
    rec_threshold = st.sidebar.slider(
        "Recognition Threshold",
        min_value=0.20,
        max_value=0.80,
        value=float(config.get("recognition_threshold", 0.45)),
        step=0.05,
        help="Lower = Stricter matching. Higher = More lenient."
    )
    
    cooldown_sec = st.sidebar.slider(
        "Attendance Cooldown (sec)",
        min_value=5,
        max_value=300,
        value=int(config.get("attendance_cooldown_seconds", 60)),
        step=5,
        help="Cooldown period before re-marking attendance for the same student."
    )

    frame_skip = st.sidebar.select_slider(
        "Process Every N Frames",
        options=[1, 2, 3, 5, 10],
        value=int(config.get("process_every_n_frames", 3)),
        help="Higher values increase FPS by skipping heavy face recognition calculations on intermediate frames."
    )

    model_name = config.get("model_name", "Facenet512")
    distance_metric = config.get("distance_metric", "cosine")

    # Fetch pre-loaded registered embeddings cache
    registered_embs = get_registered_embeddings(model_name=model_name)
    num_registered = len(registered_embs)

    if num_registered == 0:
        st.warning("⚠️ No registered students found in database. Please register students first under 'Register Student' page.")

    col_cam, col_info = st.columns([3, 2])

    with col_cam:
        st.subheader("Live Camera Stream")
        cam_placeholder = st.empty()

        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            run_cam = st.checkbox("▶️ Start Live Webcam Stream", value=False, key="run_cam_checkbox")
        with c_btn2:
            single_snapshot = st.button("📸 Capture Single Snapshot")

    with col_info:
        st.subheader("Recognition Card")
        info_placeholder = st.empty()
        log_placeholder = st.empty()

    # Initial placeholder state
    def render_idle_card():
        info_placeholder.markdown("""
            <div class="recognition-card rec-idle">
                <h3 style="color: #94A3B8; margin:0;">Camera Idle</h3>
                <p style="color: #64748B; margin-top:5px;">Check 'Start Live Webcam Stream' or take a snapshot.</p>
            </div>
        """, unsafe_allow_html=True)

    render_idle_card()

    # Handle Snapshot option or Camera Stream
    uploaded_snap = st.file_uploader("Upload Image for Recognition (Webcam Fallback)", type=["jpg", "png", "jpeg"])

    frame_to_process = None

    if uploaded_snap is not None:
        pil_img = Image.open(uploaded_snap)
        rgb_img = np.array(pil_img)
        frame_to_process = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    elif single_snapshot:
        cap = cv2.VideoCapture(config.get("camera_index", 0))
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_to_process = frame
        else:
            st.error("Failed to open physical webcam. Please ensure webcam permissions are enabled.")

    # Process frame if uploaded or snapshot taken
    if frame_to_process is not None:
        resized_frame = resize_frame(frame_to_process, target_width=config.get("frame_width", 640))
        faces = detect_faces_opencv(resized_frame)

        if len(faces) == 0:
            cam_placeholder.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            info_placeholder.markdown("""
                <div class="recognition-card rec-unknown">
                    <h3 style="color: #EF4444; margin:0;">No Face Detected</h3>
                    <p style="color: #94A3B8; margin-top:5px;">Align your face within camera view.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            for bbox in faces:
                x, y, w, h = bbox
                face_crop = resized_frame[y:y+h, x:x+w]
                live_emb = extract_face_embedding(face_crop, model_name=model_name)

                student_id, distance, confidence, is_recognized = match_face_embedding(
                    live_emb,
                    registered_embs,
                    threshold=rec_threshold,
                    metric=distance_metric
                )

                if is_recognized and student_id:
                    student_info = get_student(student_id)
                    student_name = student_info["name"] if student_info else student_id

                    # Attempt attendance recording
                    res = mark_live_attendance(student_id, confidence, cooldown_seconds=cooldown_sec)

                    draw_recognition_box(resized_frame, bbox, student_name, student_id, confidence, is_known=True)

                    status_color = "#10B981" if res["success"] else "#F59E0B"
                    info_placeholder.markdown(f"""
                        <div class="recognition-card rec-success">
                            <h2 style="color: #F8FAFC; margin:0;">{student_name}</h2>
                            <p style="color: #3B82F6; font-size:1.1rem; font-weight:600; margin:4px 0;">Student ID: {student_id}</p>
                            <p style="color: #94A3B8; margin:2px 0;">Dept: {student_info.get('department', 'N/A')}</p>
                            <h3 style="color: {status_color}; margin-top:12px;">{res['message']}</h3>
                            <p style="color: #64748B; font-size:0.85rem;">Confidence: {confidence:.1f}% | Distance: {distance:.3f}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    draw_recognition_box(resized_frame, bbox, "Unknown", "", confidence=0.0, is_known=False)
                    info_placeholder.markdown("""
                        <div class="recognition-card rec-unknown">
                            <h2 style="color: #EF4444; margin:0;">Unknown Person</h2>
                            <p style="color: #94A3B8; margin-top:6px;">Face not recognized in database.</p>
                            <p style="color: #64748B; font-size:0.85rem;">Attendance not marked.</p>
                        </div>
                    """, unsafe_allow_html=True)

            cam_placeholder.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # Continuous Live Webcam Loop
    if run_cam:
        cap = cv2.VideoCapture(config.get("camera_index", 0))
        if not cap.isOpened():
            st.error("Cannot open webcam. Please verify camera device index in Settings.")
            return

        frame_count = 0
        last_recognized_info = None

        while st.session_state.get("run_cam_checkbox", False):
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam stream disconnected.")
                break

            frame_count += 1
            resized_frame = resize_frame(frame, target_width=config.get("frame_width", 640))

            # Process heavy face recognition every N frames
            if frame_count % frame_skip == 0:
                faces = detect_faces_opencv(resized_frame)

                if len(faces) == 0:
                    last_recognized_info = {
                        "type": "none",
                        "faces": []
                    }
                else:
                    recognized_list = []
                    for bbox in faces:
                        x, y, w, h = bbox
                        face_crop = resized_frame[y:y+h, x:x+w]
                        live_emb = extract_face_embedding(face_crop, model_name=model_name)

                        student_id, distance, confidence, is_recognized = match_face_embedding(
                            live_emb,
                            registered_embs,
                            threshold=rec_threshold,
                            metric=distance_metric
                        )

                        if is_recognized and student_id:
                            student_info = get_student(student_id)
                            student_name = student_info["name"] if student_info else student_id

                            res = mark_live_attendance(student_id, confidence, cooldown_seconds=cooldown_sec)

                            recognized_list.append({
                                "bbox": bbox,
                                "known": True,
                                "name": student_name,
                                "student_id": student_id,
                                "confidence": confidence,
                                "distance": distance,
                                "department": student_info.get('department', 'N/A') if student_info else 'N/A',
                                "res": res
                            })
                        else:
                            recognized_list.append({
                                "bbox": bbox,
                                "known": False
                            })

                    last_recognized_info = {
                        "type": "faces",
                        "list": recognized_list
                    }

            # Draw overlay based on last recognized info
            if last_recognized_info and last_recognized_info.get("type") == "faces":
                item_list = last_recognized_info["list"]
                for item in item_list:
                    if item["known"]:
                        draw_recognition_box(
                            resized_frame, item["bbox"], item["name"], item["student_id"], item["confidence"], is_known=True
                        )
                    else:
                        draw_recognition_box(
                            resized_frame, item["bbox"], "Unknown", "", 0.0, is_known=False
                        )

                # Update recognition card on right panel
                primary_item = item_list[0]
                if primary_item["known"]:
                    res = primary_item["res"]
                    status_color = "#10B981" if res["success"] else "#F59E0B"
                    info_placeholder.markdown(f"""
                        <div class="recognition-card rec-success">
                            <h2 style="color: #F8FAFC; margin:0;">{primary_item['name']}</h2>
                            <p style="color: #3B82F6; font-size:1.1rem; font-weight:600; margin:4px 0;">Student ID: {primary_item['student_id']}</p>
                            <p style="color: #94A3B8; margin:2px 0;">Dept: {primary_item['department']}</p>
                            <h3 style="color: {status_color}; margin-top:12px;">{res['message']}</h3>
                            <p style="color: #64748B; font-size:0.85rem;">Confidence: {primary_item['confidence']:.1f}% | Distance: {primary_item['distance']:.3f}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    info_placeholder.markdown("""
                        <div class="recognition-card rec-unknown">
                            <h2 style="color: #EF4444; margin:0;">Unknown Person</h2>
                            <p style="color: #94A3B8; margin-top:6px;">Face not recognized in database.</p>
                            <p style="color: #64748B; font-size:0.85rem;">Attendance not marked.</p>
                        </div>
                    """, unsafe_allow_html=True)
            elif last_recognized_info and last_recognized_info.get("type") == "none":
                render_idle_card()

            cam_placeholder.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        cap.release()
