import os
import random
import datetime
import json
import numpy as np
import cv2
from src.database import (
    init_db,
    add_student,
    save_face_embedding,
    record_attendance,
    get_db_connection
)
from config import FACES_DIR, config

DEMO_STUDENTS = [
    {"student_id": "CS101", "name": "Aarav Sharma", "department": "Computer Science & Engineering", "year": "3rd Year", "section": "Section A", "email": "aarav.sharma@example.com"},
    {"student_id": "CS102", "name": "Priya Patel", "department": "Computer Science & Engineering", "year": "3rd Year", "section": "Section A", "email": "priya.patel@example.com"},
    {"student_id": "CS103", "name": "Rohan Verma", "department": "Computer Science & Engineering", "year": "3rd Year", "section": "Section B", "email": "rohan.verma@example.com"},
    {"student_id": "IT201", "name": "Ananya Roy", "department": "Information Technology", "year": "2nd Year", "section": "Section A", "email": "ananya.roy@example.com"},
    {"student_id": "IT202", "name": "Vikram Singh", "department": "Information Technology", "year": "2nd Year", "section": "Section B", "email": "vikram.singh@example.com"},
    {"student_id": "EC301", "name": "Sneha Reddy", "department": "Electronics & Communication", "year": "4th Year", "section": "Section A", "email": "sneha.reddy@example.com"},
    {"student_id": "EC302", "name": "Aditya Kumar", "department": "Electronics & Communication", "year": "4th Year", "section": "Section A", "email": "aditya.kumar@example.com"},
    {"student_id": "EE401", "name": "Kavya Nair", "department": "Electrical Engineering", "year": "1st Year", "section": "Section C", "email": "kavya.nair@example.com"},
    {"student_id": "ME501", "name": "Rahul Deshmukh", "department": "Mechanical Engineering", "year": "3rd Year", "section": "Section D", "email": "rahul.deshmukh@example.com"},
    {"student_id": "CE601", "name": "Neha Gupta", "department": "Civil Engineering", "year": "2nd Year", "section": "Section A", "email": "neha.gupta@example.com"},
    {"student_id": "CS104", "name": "Ishaan Joshi", "department": "Computer Science & Engineering", "year": "1st Year", "section": "Section B", "email": "ishaan.joshi@example.com"},
    {"student_id": "IT203", "name": "Diya Iyer", "department": "Information Technology", "year": "4th Year", "section": "Section C", "email": "diya.iyer@example.com"}
]

def create_synthetic_face_image(student_name, student_id, sample_idx):
    """Generate a synthetic face placeholder image with text for demo visualization."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Background color gradient
    color_val = (hash(student_id + str(sample_idx)) % 150) + 50
    img[:, :] = (color_val, int(color_val * 0.8), int(color_val * 1.2))
    
    # Draw face oval
    cv2.ellipse(img, (150, 140), (70, 90), 0, 0, 360, (220, 220, 240), -1)
    # Eyes
    cv2.circle(img, (120, 120), 10, (50, 50, 50), -1)
    cv2.circle(img, (180, 120), 10, (50, 50, 50), -1)
    # Smile
    cv2.ellipse(img, (150, 170), (30, 15), 0, 0, 180, (50, 50, 50), 3)
    
    # Text tag
    cv2.putText(img, student_id, (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Sample #{sample_idx+1}", (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return img

def populate_demo_data():
    """Populate database with sample students, face samples, and 30 days of attendance logs."""
    print("Initializing Database...")
    init_db()

    print("Registering Demo Students & Synthetic Embeddings...")
    for s in DEMO_STUDENTS:
        sid = s["student_id"]
        success, msg = add_student(sid, s["name"], s["department"], s["year"], s["section"], s["email"])
        if success:
            print(f" -> Added Student: {s['name']} ({sid})")

        # Create face folder
        s_folder = os.path.join(FACES_DIR, sid)
        os.makedirs(s_folder, exist_ok=True)

        # Generate 3 synthetic face samples & embeddings per student
        for idx in range(3):
            # Save synthetic image
            img = create_synthetic_face_image(s["name"], sid, idx)
            img_path = os.path.join(s_folder, f"sample_{idx+1}.jpg")
            cv2.imwrite(img_path, img)

            # Create synthetic 512D unit embedding vector
            rnd = np.random.RandomState(seed=hash(sid + str(idx)) % (2**32 - 1))
            vec = rnd.randn(512)
            vec = vec / np.linalg.norm(vec)
            save_face_embedding(sid, vec.tolist(), model_name="Facenet512")

    print("\nGenerating 30 Days Historical Attendance Records...")
    conn = get_db_connection()
    cursor = conn.cursor()

    today = datetime.date.today()
    for day_offset in range(30, -1, -1):
        record_date = today - datetime.timedelta(days=day_offset)
        # Skip Sundays for realism
        if record_date.weekday() == 6:
            continue

        date_str = record_date.strftime("%Y-%m-%d")

        # Random attendance for students on this day
        for s in DEMO_STUDENTS:
            sid = s["student_id"]
            # Student attendance probability: CS101 high (95%), ME501 lower (60%) for demo low-attendance alert
            prob = 0.60 if sid == "ME501" else (0.68 if sid == "EC302" else 0.90)

            if random.random() < prob:
                hour = random.randint(8, 10)
                minute = random.randint(0, 59)
                time_str = f"{hour:02d}:{minute:02d}:{random.randint(0, 59):02d}"
                timestamp = f"{date_str}T{time_str}"
                confidence = round(random.uniform(85.0, 98.5), 1)

                try:
                    cursor.execute("""
                        INSERT INTO attendance (student_id, date, time, timestamp, confidence, status)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (sid, date_str, time_str, timestamp, confidence, "Present"))
                except Exception as e:
                    pass

    conn.commit()
    conn.close()
    print("Demo Data Population Completed Successfully!")

if __name__ == "__main__":
    populate_demo_data()
