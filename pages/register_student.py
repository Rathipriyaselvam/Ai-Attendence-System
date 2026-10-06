import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

from config import config, FACES_DIR
from src.database import add_student, get_student
from src.face_detection import validate_single_face
from src.face_recognition import process_face_samples_for_student
from src.utils import sanitize_filename

def render_register_student_page():
    st.markdown("""
        <div class="main-header">
            <h1>👤 Student Registration & Biometric Enrollment</h1>
            <p>Register new student profiles and capture facial embeddings for AI recognition.</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize Session State for Captured Face Samples
    if "captured_samples" not in st.session_state:
        st.session_state["captured_samples"] = []

    st.subheader("1. Student Information")
    col1, col2 = st.columns(2)

    with col1:
        student_id = st.text_input("Student ID *", placeholder="e.g. CS105").strip().upper()
        name = st.text_input("Full Name *", placeholder="e.g. John Doe").strip()
        department = st.selectbox("Department *", config["departments"])

    with col2:
        year = st.selectbox("Academic Year *", config["years"])
        section = st.selectbox("Section *", config["sections"])
        email = st.text_input("Email Address (Optional)", placeholder="student@example.com").strip()

    st.markdown("---")
    st.subheader("2. Face Biometric Capture (Target: 3 - 5 Samples)")
    st.info("💡 Ensure good lighting and look straight at the camera. Exactly ONE face must be visible per capture.")

    mode_tabs = st.tabs(["📸 Webcam Capture", "📁 Upload Face Image Files"])

    with mode_tabs[0]:
        st.write("Capture face images using your webcam:")
        webcam_photo = st.camera_input("Take a Face Snapshot")

        if webcam_photo is not None:
            # Convert uploaded PIL/Streamlit image to BGR OpenCV format
            pil_image = Image.open(webcam_photo)
            rgb_arr = np.array(pil_image)
            bgr_frame = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

            is_valid, faces, msg, cropped_face = validate_single_face(bgr_frame)

            if is_valid and cropped_face is not None:
                st.session_state["captured_samples"].append(bgr_frame)
                st.success(f"✅ Sample #{len(st.session_state['captured_samples'])} captured successfully!")
            else:
                st.error(f"❌ Validation Failed: {msg}")

    with mode_tabs[1]:
        st.write("Or upload photo files from disk:")
        uploaded_files = st.file_uploader(
            "Select Face Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="file_face_uploader"
        )

        if uploaded_files:
            for file in uploaded_files:
                pil_img = Image.open(file)
                rgb_arr = np.array(pil_img)
                bgr_frame = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

                is_valid, faces, msg, cropped_face = validate_single_face(bgr_frame)
                if is_valid:
                    st.session_state["captured_samples"].append(bgr_frame)
                    st.success(f"✅ Uploaded '{file.name}' accepted as Sample #{len(st.session_state['captured_samples'])}")
                else:
                    st.error(f"❌ File '{file.name}' rejected: {msg}")

    # Display Captured Samples Gallery
    st.markdown("### Captured Samples Preview")

    num_samples = len(st.session_state["captured_samples"])

    if num_samples > 0:
        sample_cols = st.columns(min(5, num_samples))
        for idx, sample_frame in enumerate(st.session_state["captured_samples"]):
            with sample_cols[idx % 5]:
                rgb_sample = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_sample, caption=f"Sample #{idx+1}", use_column_width=True)

        if st.button("🗑️ Clear Captured Samples"):
            st.session_state["captured_samples"] = []
            st.rerun()
    else:
        st.warning("No face samples captured yet. Please capture at least 1 (recommended 3+) face samples above.")

    st.markdown("---")

    # Submit / Save Registration Button
    if st.button("💾 Save & Register Student", type="primary", use_container_width=True):
        # Input Validation
        if not student_id:
            st.error("Please enter a valid Student ID.")
            return
        if not name:
            st.error("Please enter the student's Full Name.")
            return
        if num_samples == 0:
            st.error("Please capture at least ONE face sample before registering.")
            return

        # Check existing student ID
        existing = get_student(student_id)
        if existing:
            st.error(f"Student ID '{student_id}' is already registered to '{existing['name']}'. Please use a unique ID or edit the student.")
            return

        # 1. Add student to SQLite
        success, msg = add_student(student_id, name, department, year, section, email)
        if not success:
            st.error(f"Failed to register student: {msg}")
            return

        # 2. Save face images to disk data/faces/<student_id>/
        safe_sid = sanitize_filename(student_id)
        student_dir = os.path.join(FACES_DIR, safe_sid)
        os.makedirs(student_dir, exist_ok=True)

        saved_image_paths = []
        for idx, frame in enumerate(st.session_state["captured_samples"]):
            img_path = os.path.join(student_dir, f"sample_{idx+1}.jpg")
            cv2.imwrite(img_path, frame)
            saved_image_paths.append(img_path)

        # 3. Generate DeepFace Embeddings and store in DB
        with st.spinner("Generating AI Face Embeddings using DeepFace..."):
            model_name = config.get("model_name", "Facenet512")
            successful_embs, total = process_face_samples_for_student(
                student_id=student_id,
                sample_images_paths_or_frames=saved_image_paths,
                model_name=model_name
            )

        if successful_embs > 0:
            st.balloons()
            st.success(f"🎉 Student '{name}' ({student_id}) successfully registered with {successful_embs} facial embeddings!")
            # Reset samples
            st.session_state["captured_samples"] = []
        else:
            st.warning(f"Student details saved, but facial embedding generation failed for sample images. Please re-register faces.")
