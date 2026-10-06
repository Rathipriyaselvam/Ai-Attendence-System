# 🤖 AI Attendance Analytics Dashboard

A production-style real-time AI-powered attendance management and analytics dashboard built using **Python**, **DeepFace**, **OpenCV**, **Streamlit**, **SQLite**, **Pandas**, **Plotly**, and **FPDF2**.

Demo Link:
https://ai-attendence-system-o2jentmykdusdjayqkwctg.streamlit.app/

---

## 🌟 Key Features

1. **Student Registration & Facial Sample Enrollment**
   - Register students with Student ID, Name, Department, Year, Section, and Email.
   - Live webcam capture validating that **exactly ONE face** is present in the frame.
   - Generates and stores **512-dimensional DeepFace embeddings** in SQLite.
   - Prevents duplicate Student IDs.

2. **Real-Time Webcam Face Recognition & Attendance Logging**
   - OpenCV live video stream feed with bounding box overlays and recognition tags.
   - Compare live embeddings against registered vector database using **Cosine Distance**.
   - Automatic attendance logging with **cooldown timer** and **daily duplicate prevention**.
   - Handles **Unknown Persons** cleanly without marking false attendance.

3. **Interactive Analytics Dashboard & Plotly Charts**
   - Real-time KPI summary cards (Total Students, Present Today, Absent Today, Attendance Rate, Total Records).
   - Today's Live Attendance table with department/year/section filters.
   - 6 Interactive Plotly charts:
     - Donut chart: Present vs. Absent Today
     - Line chart: Daily Attendance Trend
     - Grouped Bar chart: Department-wise Attendance
     - Bar chart: Weekly Attendance Volume
     - Horizontal Bar chart: Top 10 Attendance Performers
     - Defaulter Banner: Highlight students below 75% threshold

4. **Student Management Directory**
   - Complete directory of registered students with face sample count and thumbnails.
   - Edit student details, capture additional face samples, or delete student profiles safely (cascading database deletion and disk image cleanup).

5. **Searchable Logs & Report Exporting**
   - Filter attendance records by Date Range, Student ID, Department, Year, Section, and Status.
   - Export reports in **CSV**, **Excel (.xlsx)**, and **PDF** formats.

6. **System Configuration & Performance Tuning**
   - Switch recognition models (`Facenet512`, `Facenet`, `VGG-Face`, `ArcFace`, `OpenFace`).
   - Configure detector backends (`opencv`, `ssd`, `mtcnn`, `retinaface`), recognition threshold, cooldown seconds, frame skipping, and camera device index.

---

## 🏗️ System Architecture

```text
       Webcam Stream (OpenCV)
                 ↓
      Face Detection (Haar / OpenCV)
                 ↓
  Single Face Validation & Cropping
                 ↓
DeepFace Embedding Extraction (Facenet512)
                 ↓
  Vector Distance Comparison (Cosine)
                 ↓
  Identity Matching & Confidence Score
                 ↓
Attendance Logging (SQLite & Cooldown check)
                 ↓
 Streamlit Analytics & Plotly Visualization
```

---

## 🛠️ Tech Stack

- **Frontend / Dashboard:** Streamlit, Custom CSS
- **Computer Vision & AI:** OpenCV, DeepFace, TensorFlow, SciPy
- **Data Persistence:** SQLite3
- **Data Analytics & Charts:** Pandas, Plotly Express & Graph Objects
- **Exporting:** openpyxl (Excel), fpdf2 (PDF), CSV
- **Testing:** PyTest

---

## 📁 Directory Structure

```text
ai-attendance-dashboard/
│
├── app.py                      # Main Streamlit application entry point
├── config.py                   # Global configuration settings manager
├── create_demo_data.py         # Script to populate database with synthetic demo data
├── requirements.txt            # Project Python dependencies
├── README.md                   # Complete documentation
├── .gitignore                  # Git ignore directives
│
├── data/                       # Local SQLite database and student face images
│   ├── attendance.db
│   ├── faces/
│   └── .gitkeep
│
├── models/                     # DeepFace model embeddings cache
│   ├── embeddings/
│   └── .gitkeep
│
├── src/                        # Core Python application logic
│   ├── __init__.py
│   ├── database.py             # SQLite database layer with parameterized queries
│   ├── face_detection.py       # OpenCV face detection & bounding box overlays
│   ├── face_recognition.py     # DeepFace vector extraction & distance matching
│   ├── attendance.py           # Attendance logging business rules & cooldown
│   ├── analytics.py            # Plotly chart generation functions
│   ├── reports.py              # Report generation & PDF/Excel/CSV exports
│   └── utils.py                # Helper utilities (CSS loader, image converters)
│
├── pages/                      # Dashboard UI pages
│   ├── dashboard.py            # Main Overview KPI & Charts page
│   ├── live_attendance.py      # Real-time webcam attendance page
│   ├── register_student.py    # Student registration & biometric capture page
│   ├── students.py            # Student directory & profile management page
│   ├── attendance_records.py   # Searchable attendance log & export page
│   ├── analytics.py            # Deep analytics & defaulter tracking page
│   ├── reports.py              # Report generator page
│   └── settings.py             # System configuration settings page
│
├── tests/                      # Automated unit test suite
│   ├── test_database.py
│   ├── test_attendance.py
│   └── test_recognition.py
│
└── assets/                     # UI styling
    └── style.css
```

---

## 🚀 Installation & Setup

### 1. Clone Repository & Create Virtual Environment

```bash
git clone <repository-url>
cd "Ai Attendence system"

# Create virtual environment
python3 -m venv venv

# Activate Virtual Environment:
# On macOS / Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Populate Demo Data (Optional for Quick Demo)

To populate the database with mock students and 30 days of synthetic historical attendance data before registering real faces:

```bash
python create_demo_data.py
```

### 4. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 💡 How to Use the Application

### A. Registering a New Student
1. Open the **Register Student** page from the sidebar navigation.
2. Enter Student ID (e.g. `CS105`), Name, Department, Year, Section, and optional Email.
3. Select **📸 Webcam Capture** or upload face photo files.
4. Capture 3 to 5 face samples (ensure only 1 face is visible in the camera frame).
5. Click **Save & Register Student**. DeepFace will generate facial embeddings and save them securely.

### B. Running Live Attendance
1. Navigate to **Live Attendance**.
2. Click **▶️ Start Live Webcam Stream**.
3. Align face in camera view. Upon recognition:
   - Known student face will draw a green box showing Student Name, ID, Confidence score, and log attendance automatically.
   - Unknown person will display a red box with "Unknown Person" and skip attendance logging.
4. Cooldown settings prevent duplicate logging for the same student within specified seconds.

### C. Viewing Analytics & Exporting Reports
1. **Dashboard:** View high-level KPIs, today's log, and 6 interactive charts.
2. **Analytics:** View daily/weekly trends, department breakdowns, individual student attendance rates, and defaulters (< 75%).
3. **Reports:** Select Date Range and Department, then download official reports in **CSV**, **Excel**, or **PDF**.

---

## 🧪 Running Automated Tests

Run the PyTest test suite:

```bash
PYTHONPATH=. pytest tests/
```

All 9 unit tests cover database CRUD, duplicate prevention, cooldown enforcement, and vector distance comparison algorithms.

---

## 🔧 Troubleshooting Guide

### 1. Webcam Not Detected
- Ensure camera permissions are granted to Terminal/Python/Browser.
- Change `Camera Index` in the **Settings** page (e.g. try `0`, `1`, or `2`).
- Use the **Upload Image File** fallback on the Live Attendance page if physical webcam is unavailable.

### 2. Slow Webcam Frame Rate
- Increase **Process Every N Frames** setting in sidebar (e.g., set to `3` or `5`).
- Reduce **Processing Frame Width** in Settings (e.g., set to `480` or `640`).

### 3. Adjusting Recognition Accuracy
- If unknown faces are recognized by mistake, **lower** the Recognition Threshold in Settings (e.g. set to `0.35` or `0.40`).
- If registered students are showing as Unknown, **increase** the threshold slightly (e.g. `0.50`) or capture more face samples.

---

## 📜 License & Acknowledgments

Built for demonstrating a production-grade AI & Computer Vision engineering application using DeepFace, OpenCV, and Streamlit.
