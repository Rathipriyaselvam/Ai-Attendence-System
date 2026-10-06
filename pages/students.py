import streamlit as st
import pandas as pd
import os
import shutil
import cv2
from PIL import Image

from config import config, FACES_DIR
from src.database import (
    get_all_students,
    get_student,
    update_student,
    delete_student,
    delete_student_embeddings,
    get_embeddings_count
)
from src.face_detection import validate_single_face
from src.face_recognition import process_face_samples_for_student, get_registered_embeddings
from src.utils import sanitize_filename

def render_students_page():
    st.markdown("""
        <div class="main-header">
            <h1>👥 Student Directory & Management</h1>
            <p>View, update profiles, inspect facial samples, and manage registered students.</p>
        </div>
    """, unsafe_allow_html=True)

    students = get_all_students()

    if not students:
        st.info("No students currently registered in the database. Use 'Register Student' page to add students.")
        return

    # Convert to pandas DataFrame for table view
    df_students = pd.DataFrame(students)

    # Search & Filter bar
    search_col, dept_col = st.columns([2, 1])
    with search_col:
        search_query = st.text_input("🔍 Search by Name or Student ID", "").strip().lower()
    with dept_col:
        filter_dept = st.selectbox("Filter Department", ["All"] + config["departments"])

    # Apply filters
    filtered_df = df_students.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df["student_id"].str.lower().str.contains(search_query) |
            filtered_df["name"].str.lower().str.contains(search_query)
        ]
    if filter_dept != "All":
        filtered_df = filtered_df[filtered_df["department"] == filter_dept]

    st.write(f"Showing **{len(filtered_df)}** of **{len(df_students)}** registered student(s):")

    # Display Students Table
    display_cols = ["student_id", "name", "department", "year", "section", "email", "sample_count", "created_at"]
    rename_dict = {
        "student_id": "Student ID",
        "name": "Name",
        "department": "Department",
        "year": "Year",
        "section": "Section",
        "email": "Email",
        "sample_count": "Face Samples",
        "created_at": "Registration Date"
    }

    st.dataframe(filtered_df[display_cols].rename(columns=rename_dict), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Select Student for Management Action
    st.subheader("🛠️ Student Profile Actions")
    student_list = [f"{s['student_id']} - {s['name']}" for s in filtered_df.to_dict('records')]

    if not student_list:
        return

    selected_str = st.selectbox("Select Student to Manage", student_list)
    selected_id = selected_str.split(" - ")[0]

    student_data = get_student(selected_id)
    if not student_data:
        st.error("Student profile not found.")
        return

    action_tab1, action_tab2, action_tab3, action_tab4 = st.tabs([
        "👁️ View Profile & Face Samples",
        "✏️ Edit Profile Details",
        "📸 Add Face Samples",
        "🗑️ Delete Student"
    ])

    # Tab 1: View Profile & Face Samples
    with action_tab1:
        v_col1, v_col2 = st.columns([1, 2])
        with v_col1:
            st.markdown(f"### **{student_data['name']}**")
            st.markdown(f"**Student ID:** `{student_data['student_id']}`")
            st.markdown(f"**Department:** {student_data['department']}")
            st.markdown(f"**Year / Section:** {student_data['year']} | {student_data['section']}")
            st.markdown(f"**Email:** {student_data['email'] or 'N/A'}")
            st.markdown(f"**Registered:** {student_data['created_at'][:10]}")
            st.markdown(f"**Stored Embeddings:** {get_embeddings_count(selected_id)}")

        with v_col2:
            st.markdown("### Saved Face Samples")
            safe_sid = sanitize_filename(selected_id)
            student_dir = os.path.join(FACES_DIR, safe_sid)

            if os.path.exists(student_dir):
                sample_files = [f for f in os.listdir(student_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if sample_files:
                    sample_cols = st.columns(min(4, len(sample_files)))
                    for idx, sfile in enumerate(sample_files):
                        with sample_cols[idx % 4]:
                            img_path = os.path.join(student_dir, sfile)
                            pil_img = Image.open(img_path)
                            st.image(pil_img, caption=f"Sample #{idx+1}", use_column_width=True)
                else:
                    st.info("No face image files saved on disk for this student.")
            else:
                st.info("Face directory does not exist on disk.")

    # Tab 2: Edit Profile Details
    with action_tab2:
        with st.form(f"edit_form_{selected_id}"):
            edit_name = st.text_input("Full Name", value=student_data["name"])
            edit_dept = st.selectbox("Department", config["departments"], index=config["departments"].index(student_data["department"]) if student_data["department"] in config["departments"] else 0)
            edit_year = st.selectbox("Year", config["years"], index=config["years"].index(student_data["year"]) if student_data["year"] in config["years"] else 0)
            edit_section = st.selectbox("Section", config["sections"], index=config["sections"].index(student_data["section"]) if student_data["section"] in config["sections"] else 0)
            edit_email = st.text_input("Email", value=student_data["email"] or "")

            if st.form_submit_button("Update Details"):
                success, msg = update_student(selected_id, edit_name, edit_dept, edit_year, edit_section, edit_email)
                if success:
                    st.success("Student details updated successfully!")
                    st.rerun()
                else:
                    st.error(f"Failed to update: {msg}")

    # Tab 3: Add Face Samples
    with action_tab3:
        st.write("Capture or upload additional face samples to refine AI recognition for this student:")
        new_cam_photo = st.camera_input("Capture Additional Sample", key=f"add_cam_{selected_id}")

        if new_cam_photo is not None:
            pil_image = Image.open(new_cam_photo)
            rgb_arr = np.array(pil_image)
            bgr_frame = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

            is_valid, faces, msg, cropped_face = validate_single_face(bgr_frame)
            if is_valid:
                safe_sid = sanitize_filename(selected_id)
                student_dir = os.path.join(FACES_DIR, safe_sid)
                os.makedirs(student_dir, exist_ok=True)

                existing_count = len(os.listdir(student_dir))
                new_file_path = os.path.join(student_dir, f"sample_{existing_count+1}.jpg")
                cv2.imwrite(new_file_path, bgr_frame)

                model_name = config.get("model_name", "Facenet512")
                succ, tot = process_face_samples_for_student(selected_id, [new_file_path], model_name=model_name)

                st.success(f"✅ Added 1 new face sample for {student_data['name']}!")
                st.rerun()
            else:
                st.error(f"Validation failed: {msg}")

    # Tab 4: Delete Student
    with action_tab4:
        st.warning(f"⚠️ Warning: Deleting student '{student_data['name']}' ({selected_id}) will permanently remove all profile records, face embeddings, stored images, and attendance logs.")
        confirm_check = st.checkbox("I confirm that I want to delete this student permanently.", key=f"del_confirm_{selected_id}")

        if st.button("🔴 Permanently Delete Student", type="primary", disabled=not confirm_check):
            # 1. Delete DB student record + cascade embeddings/attendance
            success, msg = delete_student(selected_id)

            if success:
                # 2. Delete face images folder from disk
                safe_sid = sanitize_filename(selected_id)
                student_dir = os.path.join(FACES_DIR, safe_sid)
                if os.path.exists(student_dir):
                    shutil.rmtree(student_dir, ignore_errors=True)

                # 3. Reload embeddings cache
                get_registered_embeddings(force_reload=True)

                st.success(f"Student '{student_data['name']}' has been permanently deleted.")
                st.rerun()
            else:
                st.error(f"Error deleting student: {msg}")
